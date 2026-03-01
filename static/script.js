const videoElement = document.getElementById("webcam");
const predictionText = document.getElementById("prediction");


// ---------- Fullscreen Function ----------
function fullscreenVideo() {
    const video = document.getElementById("webcam");

    if (video.requestFullscreen) {
        video.requestFullscreen();
    } else if (video.webkitRequestFullscreen) { // Safari
        video.webkitRequestFullscreen();
    } else if (video.msRequestFullscreen) { // IE11
        video.msRequestFullscreen();
    }
}

// ---------- MediaPipe Hands Setup ----------
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

// ---------- Results Callback ----------
hands.onResults(async (results) => {

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {

        let data = [];

        // ✅ Extract ONLY x,y → 42 values
        results.multiHandLandmarks[0].forEach(lm => {
            data.push(lm.x, lm.y);
        });

        try {
            const res = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await res.json();

            if (result.prediction) {
                predictionText.innerText =
                    `Gesture: ${result.prediction} (${result.confidence}%)`;
            }

        } catch (err) {
            console.log("❌ Backend error:", err);
        }
    }
    else {
        predictionText.innerText = "Gesture: ---";
    }
});

// ---------- Camera Setup ----------
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});

// ---------- Start Camera ----------
camera.start();

// OPTIONAL → Auto fullscreen on start
// fullscreenVideo();