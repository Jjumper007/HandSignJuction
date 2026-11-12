import Foundation

/// Sends one recording session to `receiver.py` over plain HTTP.
///
/// Requests go out strictly one after another, in order: session start, the
/// audio chunks, session end. A failed request is retried a few times; after
/// that the queue pauses until `retry()` is called, and nothing is lost.
///
/// Plain HTTP requests through `URLSession` are the only networking that
/// watchOS allows every app to use, which is why this is not a socket stream.
///
/// All mutable state is confined to `queue`, hence `@unchecked Sendable`.
final class ChunkUploader: @unchecked Sendable {
    struct Progress {
        var sentChunks = 0
        var pendingRequests = 0
        var hasFailed = false
        var isFinished = false
        var lastError: String?
    }

    private struct Item {
        let request: URLRequest
        let body: Data
        let isChunk: Bool
        let isLast: Bool
    }

    private static let maxAttempts = 5

    private let baseURL: URL
    private let sessionID: String
    private let onProgress: (Progress) -> Void
    private let queue = DispatchQueue(label: "sonar.chunk-uploader")
    private let urlSession: URLSession

    // Everything below is touched only on `queue`.
    private var items: [Item] = []
    private var isSending = false
    private var attempts = 0
    private var nextChunkIndex = 0
    private var progress = Progress()

    /// `onProgress` is called on the main thread.
    init(baseURL: URL, sessionID: String, onProgress: @escaping (Progress) -> Void) {
        self.baseURL = baseURL
        self.sessionID = sessionID
        self.onProgress = onProgress
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 10
        configuration.waitsForConnectivity = false
        urlSession = URLSession(configuration: configuration)
    }

    /// Accepts "192.168.1.10:8765" (plain HTTP on the local network) or a full
    /// URL such as "https://example.trycloudflare.com". Returns nil when malformed.
    static func makeBaseURL(from address: String) -> URL? {
        let trimmed = address.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let text = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
        guard let url = URL(string: text), url.host != nil else { return nil }
        return url
    }

    /// Asks the receiver whether it is reachable. `completion` runs on the main thread.
    static func ping(baseURL: URL, completion: @escaping (String) -> Void) {
        var request = URLRequest(url: baseURL.appendingPathComponent("ping"))
        request.timeoutInterval = 5
        URLSession.shared.dataTask(with: request) { _, response, error in
            let result: String
            if let error = error {
                result = "Failed: \(error.localizedDescription)"
            } else if let status = (response as? HTTPURLResponse)?.statusCode, status == 200 {
                result = "Connected"
            } else {
                result = "Unexpected reply from the server"
            }
            DispatchQueue.main.async {
                completion(result)
            }
        }.resume()
    }

    /// Queues the session metadata. It is always sent before any chunk.
    func begin(metadata: [String: Any]) {
        guard let body = try? JSONSerialization.data(withJSONObject: metadata) else { return }
        let request = jsonRequest(path: "session/start")
        queue.async {
            // The request at the front may already be in flight; never displace it.
            let position = self.isSending ? 1 : 0
            self.items.insert(Item(request: request, body: body, isChunk: false, isLast: false),
                              at: min(position, self.items.count))
            self.publishAndSend()
        }
    }

    /// Queues one chunk of mono 16-bit little-endian PCM.
    func enqueueChunk(_ pcm: Data, sampleRate: Int) {
        queue.async {
            var request = URLRequest(url: self.baseURL.appendingPathComponent("chunk"))
            request.httpMethod = "POST"
            request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            request.setValue(self.sessionID, forHTTPHeaderField: "X-Session-Id")
            request.setValue(String(self.nextChunkIndex), forHTTPHeaderField: "X-Chunk-Index")
            request.setValue(String(sampleRate), forHTTPHeaderField: "X-Sample-Rate")
            self.nextChunkIndex += 1
            self.items.append(Item(request: request, body: pcm, isChunk: true, isLast: false))
            self.publishAndSend()
        }
    }

    /// Queues the end-of-session marker; the receiver then writes the WAV file.
    func finish(totalFrames: Int) {
        let request = jsonRequest(path: "session/end")
        queue.async {
            let summary: [String: Any] = [
                "session_id": self.sessionID,
                "total_chunks": self.nextChunkIndex,
                "total_frames": totalFrames,
            ]
            guard let body = try? JSONSerialization.data(withJSONObject: summary) else { return }
            self.items.append(Item(request: request, body: body, isChunk: false, isLast: true))
            self.publishAndSend()
        }
    }

    /// Resumes a queue that paused after repeated failures.
    func retry() {
        queue.async {
            self.progress.hasFailed = false
            self.attempts = 0
            self.publishAndSend()
        }
    }

    private func jsonRequest(path: String) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    private func publishAndSend() {
        publish()
        sendNext()
    }

    private func publish() {
        progress.pendingRequests = items.count
        let snapshot = progress
        let report = onProgress
        DispatchQueue.main.async {
            report(snapshot)
        }
    }

    private func sendNext() {
        guard !isSending, !progress.hasFailed, let item = items.first else { return }
        isSending = true
        let task = urlSession.uploadTask(with: item.request, from: item.body) { [weak self] _, response, error in
            guard let self = self else { return }
            self.queue.async {
                self.handleResult(response: response, error: error)
            }
        }
        task.resume()
    }

    private func handleResult(response: URLResponse?, error: Error?) {
        isSending = false
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if error == nil, (200..<300).contains(status) {
            let item = items.removeFirst()
            attempts = 0
            progress.lastError = nil
            if item.isChunk { progress.sentChunks += 1 }
            if item.isLast { progress.isFinished = true }
            publishAndSend()
            return
        }

        attempts += 1
        progress.lastError = error?.localizedDescription ?? "HTTP \(status)"
        if attempts >= ChunkUploader.maxAttempts {
            progress.hasFailed = true
            publish()
        } else {
            publish()
            queue.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                self?.sendNext()
            }
        }
    }
}
