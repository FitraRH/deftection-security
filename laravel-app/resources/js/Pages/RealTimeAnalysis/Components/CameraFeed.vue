<script setup>
import { ref, watch, onUnmounted, computed } from "vue";

// --- Props ---
const props = defineProps({
    deviceId: { type: String, default: null },
    autoScanEnabled: { type: Boolean, default: false },
    scanInterval: { type: Number, default: 1000 },
    detections: { type: Array, default: () => [] },
});

// --- Emits ---
const emit = defineEmits([
    "capture",
    "frame-for-analysis",
    "started",
    "stopped",
]);

// --- Component Refs ---
const videoRef = ref(null);
const snapshotCanvasRef = ref(null);
const boundingBoxCanvasRef = ref(null);

// --- Component State ---
const stream = ref(null);
const isCameraActive = ref(false);
const isPaused = ref(false); // NEW: State to handle pause/resume
const errorMessage = ref("");
const scanTimer = ref(null);
const isScanning = computed(
    () => props.autoScanEnabled && isCameraActive.value && !isPaused.value
);

// --- Drawing Logic ---
const drawBoundingBoxes = () => {
    const video = videoRef.value;
    const canvas = boundingBoxCanvasRef.value;
    if (!canvas || !video || !isCameraActive.value) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    props.detections.forEach((detection) => {
        const [x, y, w, h] = detection.bbox;
        const label = detection.label;
        ctx.strokeStyle =
            label.toLowerCase() === "product"
                ? "rgba(0, 255, 0, 0.8)"
                : "rgba(255, 0, 0, 0.8)";
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.font = "16px Arial";
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 20, textWidth + 10, 20);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + 5, y - 5);
    });
};

watch(() => props.detections, drawBoundingBoxes, { deep: true });

// --- Core Camera Methods ---
const startDetection = async () => {
    if (stream.value) stopDetection(false); // Clean up previous stream silently

    errorMessage.value = "";
    const constraints = {
        video: {
            deviceId: props.deviceId ? { exact: props.deviceId } : undefined,
            width: { ideal: 1280 },
            height: { ideal: 720 },
        },
    };
    try {
        stream.value = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoRef.value) {
            videoRef.value.srcObject = stream.value;
            videoRef.value.play();
            videoRef.value.onplay = () => {
                isCameraActive.value = true;
                isPaused.value = false; // Ensure not paused on start
                emit("started");
                drawBoundingBoxes();
            };
        }
    } catch (error) {
        console.error("Error accessing camera:", error);
        errorMessage.value =
            "Failed to access camera. Please check permissions.";
        isCameraActive.value = false;
    }
};

const stopDetection = (shouldEmit = true) => {
    stopScanLoop();
    if (stream.value) stream.value.getTracks().forEach((track) => track.stop());
    if (videoRef.value) videoRef.value.srcObject = null;
    const canvas = boundingBoxCanvasRef.value;
    if (canvas)
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    isCameraActive.value = false;
    isPaused.value = false; // Reset paused state on stop

    if (shouldEmit) {
        emit("stopped");
    }
};

// --- Capture & Scanning Logic ---
const pauseSession = () => {
    isPaused.value = true;
    stopScanLoop();
    console.log("Session paused.");
};

const resumeSession = () => {
    isPaused.value = false;
    startScanLoop();
    console.log("Session resumed.");
};

const saveAnalyzedFrame = async () => {
    // This function is exposed to the parent but is no longer tied to a button.
    // It remains available for programmatic use if needed.
    return new Promise((resolve) => {
        if (!isCameraActive.value) {
            resolve(null);
            return;
        }
        const canvas = snapshotCanvasRef.value;
        const video = videoRef.value;
        const overlay = boundingBoxCanvasRef.value;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => resolve(blob), "image/jpeg");
    });
};

defineExpose({ saveAnalyzedFrame });

const captureAndEmitForAnalysis = () => {
    if (!isCameraActive.value || isPaused.value) return;
    const canvas = snapshotCanvasRef.value;
    const video = videoRef.value;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
        (blob) => {
            emit("frame-for-analysis", { blob, timestamp: new Date() });
        },
        "image/jpeg",
        0.85
    );
};

const startScanLoop = () => {
    stopScanLoop();
    if (props.autoScanEnabled && props.scanInterval > 0) {
        scanTimer.value = setInterval(
            captureAndEmitForAnalysis,
            props.scanInterval
        );
    }
};

const stopScanLoop = () => {
    if (scanTimer.value) clearInterval(scanTimer.value);
    scanTimer.value = null;
};

// --- Watchers ---
watch(isScanning, (isNowScanning) => {
    if (isNowScanning) {
        startScanLoop();
    } else {
        stopScanLoop();
    }
});
watch(
    () => props.scanInterval,
    () => {
        if (isScanning.value) startScanLoop();
    }
);
watch(
    () => props.deviceId,
    (nv, ov) => {
        if (nv !== ov && isCameraActive.value) startDetection();
    }
);

onUnmounted(() => stopDetection(false));
</script>

<template>
    <div
        class="flex flex-col justify-center border border-gray-200 dark:border-dark-700 rounded-lg p-6"
    >
        <!-- Header with Title and Action Buttons -->
        <div
            class="flex flex-col items-center justify-between mb-4 sm:flex-row gap-3"
        >
            <div class="flex gap-2">
                <h2
                    class="truncate text-base font-medium tracking-wide text-gray-800 dark:text-dark-100"
                >
                    Camera Feed
                </h2>
                <!-- Live Indicator for Auto-Scanning -->
                <div v-if="isScanning" class="flex items-center ml-4 gap-1">
                    <span class="flex w-3 h-3">
                        <span
                            class="absolute inline-flex w-3 h-3 bg-red-400 rounded-full opacity-75 animate-ping"
                        ></span>
                        <span
                            class="relative inline-flex w-3 h-3 bg-red-500 rounded-full"
                        ></span>
                    </span>
                    <span class="text-sm text-red-300">Scanning...</span>
                </div>
            </div>
            <div class="flex space-x-2">
                <!-- UPDATED: Dynamic Start/Pause/Resume Button -->
                <button
                    v-if="!isCameraActive"
                    @click="startDetection"
                    class="btn-base btn gap-2 this:success bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90"
                >
                    <font-awesome-icon icon="fa-solid fa-play" />
                    Start Session
                </button>
                <button
                    v-else-if="!isPaused"
                    @click="pauseSession"
                    class="btn-base btn gap-2 this:warning bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90"
                >
                    <font-awesome-icon icon="fa-solid fa-pause" />
                    Pause Session
                </button>
                <button
                    v-else
                    @click="resumeSession"
                    class="btn-base btn gap-2 this:success bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90"
                >
                    <font-awesome-icon icon="fa-solid fa-play" />
                    Resume
                </button>

                <!-- Stop Button -->
                <button
                    @click="stopDetection(true)"
                    :disabled="!isCameraActive"
                    class="btn-base btn gap-2 this:error bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                >
                    <font-awesome-icon icon="fa-solid fa-stop" />
                    Stop Session
                </button>
            </div>
        </div>

        <!-- Video Display Area -->
        <div
            class="relative w-full overflow-hidden bg-black rounded-md aspect-video"
        >
            <video
                ref="videoRef"
                class="w-full h-full"
                muted
                playsinline
            ></video>

            <!-- Bounding Box Canvas Overlay -->
            <canvas
                ref="boundingBoxCanvasRef"
                class="absolute top-0 left-0 w-full h-full"
            >
            </canvas>

            <!-- Paused State Overlay -->
            <div
                v-if="isPaused"
                class="absolute inset-0 flex flex-col items-center justify-center p-4 text-center bg-black bg-opacity-70 backdrop-blur-sm"
            >
                <font-awesome-icon
                    icon="fa-solid fa-pause"
                    size="3x"
                    class="text-white"
                />
                <p class="mt-4 text-2xl font-medium text-white">
                    Session Paused
                </p>
            </div>

            <!-- Placeholder shown when the camera is off -->
            <div
                v-if="!isCameraActive"
                class="absolute inset-0 flex flex-col items-center justify-center p-4 text-center bg-gray-200 dark:bg-dark-800 bg-opacity-80"
            >
                <font-awesome-icon icon="fa-solid fa-video-slash" size="2xl" />
                <p
                    class="mt-4 mb-2 text-lg font-medium text-gray-700 dark:text-dark-100"
                >
                    Camera feed will appear here
                </p>
                <p v-if="!errorMessage">Click "Start Session" to begin</p>
                <!-- Display error message if camera access fails -->
                <p
                    v-if="errorMessage"
                    class="max-w-md mt-2 text-sm text-red-400"
                >
                    {{ errorMessage }}
                </p>
            </div>
        </div>

        <!-- Hidden canvas used for capturing frames -->
        <canvas ref="snapshotCanvasRef" class="hidden"></canvas>
    </div>
</template>
