<script setup>
import { ref, computed } from "vue";

// --- Props ---
const props = defineProps({
    summary: {
        type: Object,
        default: () => ({
            imageName: "work.png",
            originalSize: "16x16",
            analysisDate: "09/07/2025, 09:43:49",
            scannedBy: "User Name", // Added scanned by info
            finalDecision: "GOOD",
            anomalyScore: 0.6734,
            anomalyConfidenceLevel: "Medium", // Changed from confidenceLevel
            status: "Completed",
            // Removed processingTime from here, moved to performance card
            // Performance metrics moved to separate card
            totalProcessingTime: "2.421s",
            preprocessingTime: "0.483s",
            anomalyInferenceTime: "0.902s",
            classificationInferenceTime: "0.119s", // Only show if defect
            postprocessingTime: "0.918s",
        }),
    },
});

// --- Computed Properties ---
const confidenceTextClass = computed(() => {
    const level = props.summary.anomalyConfidenceLevel?.toLowerCase();
    if (level === "very high" || level === "high")
        return { score: "high", color: "success" };
    if (level === "medium") return { score: "medium", color: "warning" };
    return { score: "low", color: "error" };
});

const isDefectScan = computed(() => {
    return props.summary.finalDecision === "DEFECT";
});
</script>

<template>
    <div class="gap-6 grid grid-cols-1 lg:grid-cols-3">
        <!-- Card 1: Image Information (Updated) -->
        <div
            class="bg-white dark:bg-dark-800 shadow-sm dark:shadow-none px-6 py-5 border border-gray-200 dark:border-none rounded-lg"
        >
            <h3
                class="flex items-center gap-4 mb-6 font-semibold text-gray-900 dark:text-dark-50 text-lg"
            >
                <font-awesome-icon icon="fa-solid fa-image" />
                Image Information
            </h3>
            <div class="space-y-5">
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Image Name</span
                    >
                    <span class="text-right">{{ summary.imageName }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Analysis Date</span
                    >
                    <span class="text-right">{{ summary.analysisDate }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Scanned By</span
                    >
                    <span class="text-right">{{
                        summary.scannedBy || "Unknown"
                    }}</span>
                </div>
            </div>
        </div>

        <!-- Card 2: Detection Results (Updated) -->
        <div
            class="bg-white dark:bg-dark-800 shadow-sm dark:shadow-none px-6 py-5 border border-gray-200 dark:border-none rounded-lg"
        >
            <h3
                class="flex items-center gap-4 mb-6 font-semibold text-gray-900 dark:text-dark-50 text-lg"
            >
                <font-awesome-icon icon="fa-solid fa-magnifying-glass" />
                Detection Results
            </h3>
            <div class="space-y-4 text-sm">
                <div class="flex justify-between items-center">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Final Decision</span
                    >
                    <div
                        v-if="summary.finalDecision === 'GOOD'"
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:success"
                    >
                        {{ summary.finalDecision }}
                    </div>
                    <div
                        v-else
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:error"
                    >
                        DEFECT
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Anomaly Score</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter uppercase badge-base badge"
                        :class="`this:${confidenceTextClass.color}`"
                    >
                        {{ summary.anomalyScore?.toFixed(4) || "0.0000" }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Anomaly Confidence</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter uppercase badge-base badge"
                        :class="`this:${confidenceTextClass.color}`"
                    >
                        {{ summary.anomalyConfidenceLevel || "Unknown" }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Status</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter uppercase badge-base badge this:success"
                    >
                        {{ summary.status }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Card 3: Processing Performance (Updated) -->
        <div
            class="bg-white dark:bg-dark-800 shadow-sm dark:shadow-none px-6 py-5 border border-gray-200 dark:border-none rounded-lg"
        >
            <h3
                class="flex items-center gap-4 mb-6 font-semibold text-gray-900 dark:text-dark-50 text-lg"
            >
                <font-awesome-icon icon="fa-solid fa-clock" />
                Processing Performance
            </h3>
            <div class="space-y-4 text-sm">
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Total Time</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:info"
                    >
                        {{ summary.totalProcessingTime || "0.000s" }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Preprocessing</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:secondary"
                    >
                        {{ summary.preprocessingTime || "0.000s" }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Anomaly Inference</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:secondary"
                    >
                        {{ summary.anomalyInferenceTime || "0.000s" }}
                    </div>
                </div>
                <div v-if="isDefectScan" class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Classification</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:secondary"
                    >
                        {{ summary.classificationInferenceTime || "0.000s" }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Postprocessing</span
                    >
                    <div
                        class="bg-this-darker/[0.07] dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:secondary"
                    >
                        {{ summary.postprocessingTime || "0.000s" }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
