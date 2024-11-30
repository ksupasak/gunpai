import os

let log = OSLog(subsystem: "com.yourapp.model", category: "CoreML")

os_log("Starting to load model", log: log, type: .info)

// Load model asynchronously
model.load(completionHandler: { error in
    if let error = error {
        os_log("Error loading model: %@", log: log, type: .error, "\(error)")
    } else {
        os_log("Model loaded successfully!", log: log, type: .info)
    }
})