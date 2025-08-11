// Add these changes to your RealTimeAnalysis/Index.vue

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from "vue";
import { useForm } from "@inertiajs/vue3";
import axios from "axios";

// --- Import all the child components ---
import CameraFeed from "./Components/CameraFeed.vue";
import SessionControls from "./Components/SessionControls.vue";
import LiveStatistics from "./Components/LiveStatistics.vue";
import ScreenshotGallery from "./Components/ScreenshotGallery.vue";
import RecentDetections from "./Components/RecentDetections.vue";
import SessionEndModal from "./Components/Modals/SessionEndModal.vue";

// --- Component Refs ---
const cameraFeedRef = ref(null);

// --- Session Management State ---
const currentSession = ref(null);
const sessionLoading = ref(false);
const sessionError = ref(null);
const flaskApiStatus = ref(null);

// --- Session Configuration Form ---
const sessionForm = useForm({
    camera_location: "",
    session_config: {},
});

// --- State Management ---
const sessionConfig = reactive({
    selectedCameraId: null,
    scanInterval: 1500,
    autoCaptureEnabled: true,
});

const currentDetections = ref([]);
const stats = reactive({
    totalFrames: 0,
    goodProducts: 0,
    defectiveProducts: 0,
    screenshots: 0,
});
const allScreenshots = ref([]);

// --- Modal State ---
const isModalVisible = ref(false);
const modalState = ref("processing"); // 'processing' or 'finished'

// --- Session Management Functions ---
const startSession = async () => {
    sessionLoading.value = true;
    sessionError.value = null;

    try {
        sessionForm.camera_location =
            sessionConfig.selectedCameraId || "Default Camera";
        sessionForm.session_config = {
            scanInterval: sessionConfig.scanInterval,
            autoCaptureEnabled: sessionConfig.autoCaptureEnabled,
        };

        const response = await axios.post("/api/realtime/sessions/start", {
            camera_location: sessionForm.camera_location,
            session_config: sessionForm.session_config,
        });

        const data = response.data;

        if (data.success) {
            currentSession.value = data.session;
            flaskApiStatus.value = data.flask_api;
            resetSession();
            console.log("Session started successfully:", data.message);
            console.log("Session ID:", data.session.id);
        } else {
            throw new Error(data.message || "Failed to start session");
        }
    } catch (error) {
        sessionError.value = error.response?.data?.message || error.message;
        console.error("Error starting session:", error);
    } finally {
        sessionLoading.value = false;
    }
};

const pauseSession = async () => {
    if (!currentSession.value) return;

    sessionLoading.value = true;
    try {
        const response = await axios.post("/api/realtime/sessions/pause");
        currentSession.value = response.data.session;
        console.log("Session paused successfully");
    } catch (error) {
        sessionError.value = error.response?.data?.message || error.message;
    } finally {
        sessionLoading.value = false;
    }
};

const resumeSession = async () => {
    if (!currentSession.value) return;

    sessionLoading.value = true;
    try {
        const response = await axios.post("/api/realtime/sessions/resume");
        currentSession.value = response.data.session;
        console.log("Session resumed successfully");
    } catch (error) {
        sessionError.value = error.response?.data?.message || error.message;
    } finally {
        sessionLoading.value = false;
    }
};

const stopSession = async () => {
    console.log("stopSession called");
    console.log("Current session:", currentSession.value);

    if (!currentSession.value) {
        console.warn("No session to stop");
        return;
    }

    sessionLoading.value = true;
    sessionError.value = null;

    try {
        console.log("Sending stop request...");

        const response = await axios.post("/api/realtime/sessions/stop");

        console.log("Stop response:", response);
        console.log("Stop response data:", response.data);

        const data = response.data;

        if (data.success) {
            const completedSession = data.session;
            currentSession.value = null;

            // Update modal with final session stats
            modalState.value = "finished";
            isModalVisible.value = true;

            console.log("Session stopped successfully:", data.message);
            console.log("Final Session Stats:", completedSession);
        } else {
            throw new Error(data.message || "Failed to stop session");
        }
    } catch (error) {
        console.error("Error stopping session:", error);
        console.error("Error response:", error.response);
        console.error("Error response data:", error.response?.data);

        sessionError.value = error.response?.data?.message || error.message;

        // If it's a 404, it means no session was found - clear the current session
        if (error.response?.status === 404) {
            console.log("Session not found, clearing current session");
            currentSession.value = null;
        }
    } finally {
        sessionLoading.value = false;
    }
};

const getCurrentSession = async () => {
    try {
        const response = await axios.get("/api/realtime/sessions/current");
        currentSession.value = response.data.session;

        if (response.data.session) {
            console.log("Found existing session:", response.data.session.id);
            stats.totalFrames = response.data.session.total_frames_processed;
            stats.goodProducts = response.data.session.good_count;
            stats.defectiveProducts = response.data.session.defect_count;
        }
    } catch (error) {
        console.error("Error getting current session:", error);
    }
};

// --- Computed Properties ---
const detectionRate = computed(() => {
    if (stats.totalFrames === 0) return 0;
    return stats.defectiveProducts / stats.totalFrames;
});

const recentDetections = computed(() => {
    return allScreenshots.value.filter((s) => s.type === "Defect").slice(0, 5);
});

const sessionDuration = computed(() => {
    if (!currentSession.value) return "00:00:00";

    const start = new Date(currentSession.value.session_start);
    const now = new Date();
    const diff = Math.floor((now - start) / 1000);

    const hours = Math.floor(diff / 3600);
    const minutes = Math.floor((diff % 3600) / 60);
    const seconds = diff % 60;

    return `${hours.toString().padStart(2, "0")}:${minutes
        .toString()
        .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
});

const isSessionActive = computed(() => {
    return (
        currentSession.value && currentSession.value.session_status === "active"
    );
});

const isSessionPaused = computed(() => {
    return (
        currentSession.value && currentSession.value.session_status === "paused"
    );
});

const hasActiveOrPausedSession = computed(() => {
    return (
        currentSession.value &&
        ["active", "paused"].includes(currentSession.value.session_status)
    );
});

const sessionStatusColor = computed(() => {
    if (!currentSession.value) return "";

    switch (currentSession.value.session_status) {
        case "active":
            return "text-green-600";
        case "paused":
            return "text-yellow-600";
        case "completed":
            return "text-blue-600";
        case "aborted":
            return "text-red-600";
        default:
            return "text-gray-600";
    }
});

const sessionStatusText = computed(() => {
    if (!currentSession.value) return "";

    switch (currentSession.value.session_status) {
        case "active":
            return "Active Session";
        case "paused":
            return "Paused Session";
        case "completed":
            return "Completed Session";
        case "aborted":
            return "Aborted Session";
        default:
            return "Unknown Status";
    }
});

// --- Main Analysis Logic (updated with session tracking) ---
const handleFrameForAnalysis = async (frameData) => {
    // Only process if we have an active session
    if (!isSessionActive.value) {
        console.warn("No active session - frame analysis skipped");
        return;
    }

    // 1. Simulate API call
    const fakeApiResponse = await simulateApiCall();

    // 2. Pass detection data down to CameraFeed to draw the boxes
    currentDetections.value = fakeApiResponse.detections;

    // 3. Wait for the next frame to ensure boxes are drawn
    await new Promise((resolve) => requestAnimationFrame(resolve));

    // 4. Call the child component to get a NEW screenshot that includes the overlay
    if (cameraFeedRef.value) {
        const blobWithOverlay = await cameraFeedRef.value.saveAnalyzedFrame();

        if (blobWithOverlay) {
            // 5. Process the result using the NEW blob
            stats.totalFrames++;
            if (fakeApiResponse.status === "good") {
                stats.goodProducts++;
                addScreenshot(blobWithOverlay, "Good", frameData.timestamp);
            } else if (fakeApiResponse.status === "defected") {
                stats.defectiveProducts++;
                addScreenshot(blobWithOverlay, "Defect", frameData.timestamp);
            }

            // 6. Here you would also send this data to your backend to update the session
            // updateSessionStats();
        }
    }

    // 7. Clear bounding boxes after a delay
    setTimeout(() => {
        currentDetections.value = [];
    }, 2500);
};

const simulateApiCall = () => {
    return new Promise((resolve) => {
        setTimeout(() => {
            const isDefect = Math.random() > 0.6; // 40% chance of defect
            resolve({
                status: isDefect ? "defected" : "good",
                detections: isDefect
                    ? [
                          { label: "Product", bbox: [150, 200, 400, 250] },
                          { label: "Scratch", bbox: [250, 280, 150, 20] },
                      ]
                    : [{ label: "Product", bbox: [150, 200, 400, 250] }],
            });
        }, 500); // Simulate 500ms network latency
    });
};

// --- Event Handlers (updated with session management) ---
const handleDetectionStarted = async () => {
    console.log("Detection starting...");

    if (!isSessionActive.value) {
        console.log("No active session. Starting new session...");
        await startSession();
    }

    if (isSessionActive.value) {
        console.log(
            "Detection started with active session:",
            currentSession.value.id
        );
    }
};

const handleDetectionStopped = async () => {
    console.log("Detection stopped. Current session:", currentSession.value);
    console.log("Is session active?", isSessionActive.value);

    if (isSessionActive.value || isSessionPaused.value) {
        isModalVisible.value = true;
        modalState.value = "processing";

        // Simulate a delay for processing the session report
        setTimeout(async () => {
            console.log("Processing timeout reached, calling stopSession");
            await stopSession();
        }, 2000); // 2-second delay
    } else {
        console.log("No active/paused session, just showing modal");
        // If no active session, just show the modal with current stats
        isModalVisible.value = true;
        modalState.value = "finished";
    }
};

const handleManualCapture = (screenshotData) => {
    addScreenshot(screenshotData.blob, "Manual", screenshotData.timestamp);
};

// --- Modal Action Handlers ---
const handleStartNewSession = async () => {
    resetSession();
    isModalVisible.value = false;
    await startSession();
};

const handleViewDetails = () => {
    console.log("Navigating to session details page...");
    // Navigate to session history or details page
    window.location.href = "/analysis/session-history";
};

// --- Helper Functions ---
const addScreenshot = (blob, type, timestamp) => {
    const screenshot = {
        id: `${type}-${timestamp.getTime()}`,
        url: URL.createObjectURL(blob),
        timestamp: timestamp,
        type: type,
    };
    allScreenshots.value.unshift(screenshot);
    stats.screenshots++;
};

const resetSession = () => {
    stats.totalFrames = 0;
    stats.goodProducts = 0;
    stats.defectiveProducts = 0;
    stats.screenshots = 0;
    allScreenshots.value.forEach((img) => URL.revokeObjectURL(img.url));
    allScreenshots.value = [];
    currentDetections.value = [];
    isModalVisible.value = false;
    sessionError.value = null;
    console.log("Session stats have been reset.");
};

// --- Lifecycle hooks ---
onMounted(async () => {
    // Check for existing active session on page load
    await getCurrentSession();
});

onUnmounted(() => {
    // Clean up screenshot URLs
    allScreenshots.value.forEach((img) => URL.revokeObjectURL(img.url));
});
</script>

<template>
    <!-- Session Status Bar -->
    <div
        v-if="isSessionActive || sessionLoading || sessionError"
        class="mb-6 p-4 border rounded-lg"
        :class="{
            'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800':
                isSessionActive,
            'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800':
                sessionLoading,
            'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800':
                sessionError,
        }"
    >
        <!-- Active Session -->
        <div v-if="isSessionActive" class="flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="flex items-center space-x-2">
                    <div
                        class="bg-green-500 rounded-full w-3 h-3 animate-pulse"
                    ></div>
                    <span
                        class="font-medium text-green-800 dark:text-green-200"
                    >
                        {{ sessionStatusText }}
                    </span>
                </div>
                <div class="text-green-600 dark:text-green-300 text-sm">
                    ID: {{ currentSession.id }} | Duration:
                    {{ sessionDuration }}
                </div>
            </div>
            <div class="flex space-x-2">
                <button
                    @click="pauseSession"
                    :disabled="sessionLoading"
                    class="bg-yellow-600 hover:bg-yellow-700 disabled:opacity-50 px-3 py-2 rounded-md text-white text-sm disabled:cursor-not-allowed"
                >
                    {{ sessionLoading ? "Pausing..." : "Pause" }}
                </button>
                <button
                    @click="stopSession"
                    :disabled="sessionLoading"
                    class="bg-red-600 hover:bg-red-700 disabled:opacity-50 px-3 py-2 rounded-md text-white text-sm disabled:cursor-not-allowed"
                >
                    {{ sessionLoading ? "Stopping..." : "Stop" }}
                </button>
            </div>
        </div>

        <!-- Paused Session -->
        <div
            v-else-if="isSessionPaused"
            class="flex justify-between items-center"
        >
            <div class="flex items-center space-x-3">
                <div class="flex items-center space-x-2">
                    <div class="bg-yellow-500 rounded-full w-3 h-3"></div>
                    <span
                        class="font-medium text-yellow-800 dark:text-yellow-200"
                    >
                        {{ sessionStatusText }}
                    </span>
                </div>
                <div class="text-yellow-600 dark:text-yellow-300 text-sm">
                    ID: {{ currentSession.id }} | Duration:
                    {{ sessionDuration }}
                </div>
            </div>
            <div class="flex space-x-2">
                <button
                    @click="resumeSession"
                    :disabled="sessionLoading"
                    class="bg-green-600 hover:bg-green-700 disabled:opacity-50 px-3 py-2 rounded-md text-white text-sm disabled:cursor-not-allowed"
                >
                    {{ sessionLoading ? "Resuming..." : "Resume" }}
                </button>
                <button
                    @click="stopSession"
                    :disabled="sessionLoading"
                    class="bg-red-600 hover:bg-red-700 disabled:opacity-50 px-3 py-2 rounded-md text-white text-sm disabled:cursor-not-allowed"
                >
                    {{ sessionLoading ? "Stopping..." : "Stop" }}
                </button>
            </div>
        </div>

        <!-- Loading -->
        <div v-else-if="sessionLoading" class="flex items-center space-x-3">
            <div
                class="border-2 border-yellow-600 border-t-transparent rounded-full w-5 h-5 animate-spin"
            ></div>
            <span class="font-medium text-yellow-800 dark:text-yellow-200">
                Managing session...
            </span>
        </div>

        <!-- Error -->
        <div v-else-if="sessionError" class="flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <svg
                    class="w-5 h-5 text-red-600"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                >
                    <path
                        fill-rule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clip-rule="evenodd"
                    ></path>
                </svg>
                <span class="font-medium text-red-800 dark:text-red-200">
                    Session Error: {{ sessionError }}
                </span>
            </div>
            <button
                @click="sessionError = null"
                class="text-red-600 hover:text-red-700 text-sm underline"
            >
                Dismiss
            </button>
        </div>
    </div>

    <!-- No Active Session Prompt -->
    <div
        v-if="!hasActiveOrPausedSession && !sessionLoading && !sessionError"
        class="bg-blue-50 dark:bg-blue-900/20 mb-6 p-4 border border-blue-200 dark:border-blue-800 rounded-lg"
    >
        <div class="flex justify-between items-center">
            <div>
                <h3 class="font-medium text-blue-800 dark:text-blue-200">
                    Ready to Start Detection
                </h3>
                <p class="mt-1 text-blue-600 dark:text-blue-300 text-sm">
                    Start a new session to begin real-time defect detection
                </p>
            </div>
            <button
                @click="startSession"
                class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md text-white text-sm"
            >
                Start Session
            </button>
        </div>
    </div>
    <!-- Main Content -->
    <div class="relative w-full">
        <div class="gap-6 grid grid-cols-1 lg:grid-cols-3">
            <!-- Main Content: Camera Feed -->
            <div class="lg:col-span-2">
                <CameraFeed
                    ref="cameraFeedRef"
                    :device-id="sessionConfig.selectedCameraId"
                    :auto-scan-enabled="sessionConfig.autoCaptureEnabled"
                    :scan-interval="sessionConfig.scanInterval"
                    :detections="currentDetections"
                    :session-active="isSessionActive"
                    @frame-for-analysis="handleFrameForAnalysis"
                    @capture="handleManualCapture"
                    @started="handleDetectionStarted"
                    @stopped="handleDetectionStopped"
                />
            </div>

            <!-- Sidebar: Controls and Stats -->
            <div class="flex flex-col justify-between">
                <LiveStatistics
                    :total-frames="stats.totalFrames"
                    :good-products="stats.goodProducts"
                    :defective-products="stats.defectiveProducts"
                    :detection-rate="detectionRate"
                    :screenshots="stats.screenshots"
                    :session-duration="sessionDuration"
                    :session-id="currentSession?.id"
                />

                <div
                    class="bg-gray-200 dark:bg-dark-500 max-2xl:my-6 h-px"
                ></div>

                <SessionControls
                    :session-active="isSessionActive"
                    :session-loading="sessionLoading"
                    @update:cameraId="
                        (id) => (sessionConfig.selectedCameraId = id)
                    "
                    @update:scanInterval="
                        (val) => (sessionConfig.scanInterval = val)
                    "
                    @update:autoCapture="
                        (val) => (sessionConfig.autoCaptureEnabled = val)
                    "
                    @reset-stats="resetSession"
                    @start-session="startSession"
                    @pause-session="pauseSession"
                    @resume-session="resumeSession"
                    @stop-session="stopSession"
                />
            </div>
        </div>

        <!-- Screenshot Gallery & Recent Detections -->
        <div class="gap-6 grid grid-cols-1 lg:grid-cols-3 mt-6">
            <RecentDetections :detections="recentDetections" />
            <div class="lg:col-span-2">
                <ScreenshotGallery
                    :screenshots="allScreenshots"
                    @clear-all="resetSession"
                />
            </div>
        </div>

        <!-- Session End Modal -->
        <SessionEndModal
            :is-visible="isModalVisible"
            :state="modalState"
            :stats="stats"
            :session="currentSession"
            @start-new="handleStartNewSession"
            @view-details="handleViewDetails"
            @close="isModalVisible = false"
        />
    </div>
</template>
