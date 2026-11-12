import AVFoundation

/// A linear up-chirp that is repeated back to back with no gap.
///
/// Must stay sample-for-sample identical to `make_chirp` in `sonar/chirp.py`.
/// If you change one, change both.
struct ChirpSpec {
    let startFrequency: Double
    let endFrequency: Double
    let samples: Int
    let sampleRate: Double
    let taperSamples: Int

    init(band: ChirpBand, sampleRate: Double) {
        startFrequency = band.startFrequency
        endFrequency = band.endFrequency
        samples = Int((AppConfig.chirpDurationSeconds * sampleRate).rounded())
        self.sampleRate = sampleRate
        taperSamples = AppConfig.chirpTaperSamples
    }
}

enum ChirpGenerator {
    /// One chirp period, in the range [-1, 1].
    static func makeChirp(_ spec: ChirpSpec) -> [Float] {
        let duration = Double(spec.samples) / spec.sampleRate
        let sweepRate = (spec.endFrequency - spec.startFrequency) / duration
        var chirp = [Float](repeating: 0, count: spec.samples)
        for n in 0..<spec.samples {
            let t = Double(n) / spec.sampleRate
            let phase = 2.0 * Double.pi * (spec.startFrequency * t + 0.5 * sweepRate * t * t)
            chirp[n] = Float(taper(spec, position: n) * sin(phase))
        }
        return chirp
    }

    /// A buffer holding `periods` chirps, ready to be looped by an `AVAudioPlayerNode`.
    static func makeLoopBuffer(spec: ChirpSpec, amplitude: Float, periods: Int,
                               format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frameCount = spec.samples * periods
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(frameCount)),
              let channels = buffer.floatChannelData else {
            return nil
        }
        let chirp = makeChirp(spec)
        for channel in 0..<Int(format.channelCount) {
            let samples = channels[channel]
            for frame in 0..<frameCount {
                samples[frame] = amplitude * chirp[frame % spec.samples]
            }
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        return buffer
    }

    private static func taper(_ spec: ChirpSpec, position n: Int) -> Double {
        let m = spec.taperSamples
        guard m > 0 else { return 1.0 }
        if n < m {
            return 0.5 * (1.0 - cos(Double.pi * (Double(n) + 0.5) / Double(m)))
        }
        if n > spec.samples - 1 - m {
            return 0.5 * (1.0 - cos(Double.pi * (Double(spec.samples - n) - 0.5) / Double(m)))
        }
        return 1.0
    }
}
