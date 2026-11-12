import Foundation

/// Measures how much of the microphone signal lies in the chirp band.
///
/// A 4th-order Butterworth high-pass just below the chirp isolates the band;
/// comparing its output power before and after the speaker starts gives an
/// on-watch answer to "does the microphone hear the chirp?" that needs no network.
struct BandLevelMeter {
    private var sections: [Biquad]
    let isUsable: Bool

    init(cutoffFrequency: Double, sampleRate: Double) {
        isUsable = cutoffFrequency < 0.45 * sampleRate
        let qualityFactors = [0.541_196_10, 1.306_562_96]   // 4th-order Butterworth
        sections = qualityFactors.map {
            Biquad.highPass(cutoff: cutoffFrequency, quality: $0, sampleRate: sampleRate)
        }
    }

    /// Returns the mean power of the whole signal and of the chirp band, for
    /// samples scaled to [-1, 1].
    mutating func process(_ samples: [Int16]) -> (total: Double, band: Double) {
        guard !samples.isEmpty else { return (0, 0) }
        var total = 0.0
        var band = 0.0
        for sample in samples {
            let x = Double(sample) / 32768.0
            total += x * x
            var y = x
            for index in sections.indices {
                y = sections[index].process(y)
            }
            band += y * y
        }
        let count = Double(samples.count)
        return (total / count, isUsable ? band / count : 0)
    }

    static func decibels(_ power: Double) -> Double {
        10.0 * log10(max(power, 1e-12))
    }
}

/// One second-order filter section (transposed direct form II).
private struct Biquad {
    let b0, b1, b2, a1, a2: Double
    var z1 = 0.0
    var z2 = 0.0

    static func highPass(cutoff: Double, quality: Double, sampleRate: Double) -> Biquad {
        let omega = 2.0 * Double.pi * cutoff / sampleRate
        let alpha = sin(omega) / (2.0 * quality)
        let cosine = cos(omega)
        let a0 = 1.0 + alpha
        return Biquad(b0: (1.0 + cosine) / 2.0 / a0,
                      b1: -(1.0 + cosine) / a0,
                      b2: (1.0 + cosine) / 2.0 / a0,
                      a1: -2.0 * cosine / a0,
                      a2: (1.0 - alpha) / a0)
    }

    mutating func process(_ x: Double) -> Double {
        let y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        return y
    }
}
