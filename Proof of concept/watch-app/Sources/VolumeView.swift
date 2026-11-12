import SwiftUI
import WatchKit

/// Lets the Digital Crown set the speaker volume used by this app.
///
/// The chirp is silent to most adults, so there is no way to hear whether the
/// volume is up. Open this screen once and turn the crown to the maximum.
struct VolumeView: View {
    var body: some View {
        VStack(spacing: 8) {
            VolumeControl()
                .frame(height: 60)
            Text("Turn the Digital Crown to raise the volume to the maximum.")
                .font(.footnote)
                .multilineTextAlignment(.center)
        }
        .navigationTitle("Volume")
    }
}

/// SwiftUI wrapper around the WatchKit volume control.
private struct VolumeControl: WKInterfaceObjectRepresentable {
    func makeWKInterfaceObject(context: Context) -> WKInterfaceVolumeControl {
        let control = WKInterfaceVolumeControl(origin: .local)
        DispatchQueue.main.async {
            control.focus()
        }
        return control
    }

    func updateWKInterfaceObject(_ control: WKInterfaceVolumeControl, context: Context) {}
}
