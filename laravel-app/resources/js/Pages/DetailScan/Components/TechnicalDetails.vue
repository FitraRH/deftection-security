<script setup>
import { computed } from "vue";

// --- Props ---
const props = defineProps({
    details: {
        type: Object,
        required: true,
        default: () => ({
            parameters: {
                "Total Processing Time": "2.421s",
                "Image Preprocessing": "0.483s",
                "Anomaly Inference": "0.902s",
                "Classification Inference": "0.119s", // Only for defects
                Postprocessing: "0.918s",
                "Model Used": "HRNet + Anomalib",
            },
            scanMetrics: {
                "Anomaly Score": 0.862,
                "Anomaly Confidence Level": "Low",
                "Anomaly Threshold": 0.622,
                "Classification Avg Confidence": 0.053, // Only for defects
            },
            threatData: {
                status: "SUSPICIOUS",
                riskLevel: "MEDIUM",
                fileHash:
                    "7ca498a8ff1d2a19c392db95ae52c0f03e46f4caf4ee7fc905835c87116cddd7",
                scanTime: "842.4ms",
                securityFlags: [
                    "Voluptatem Numquam",
                    "Vero",
                    "Enim",
                    "Veniam Et",
                    "Rerum",
                ],
                detailedAnalysis: {
                    "Aut Sapiente":
                        "Non error est blanditiis cum id consequatur dolorem.",
                },
                possibleAttacks: [
                    "Ut commodi atque pariatur et. Est ad ipsum corrupti veniam doloribus sed.",
                    "Quo voluptatem facere non voluptas voluptas dolore aut. Vitae optio error ut enim necessitatibus ab.",
                ],
            },
            isDefect: false,
            hasThreat: false,
            rawData: {
                analysis_date: "2025-07-30 12:03:00",
                anomaly_score: 0.862,
                confidence_level: "Low",
                final_decision: "DEFECTIVE",
                id: 27,
                image_name: "voluptatem.png",
                defects: ["discoloration", "missing_part"],
            },
        }),
    },
});

// --- Computed Properties ---
const formattedRawData = computed(() => {
    return JSON.stringify(props.details.rawData, null, 2);
});

// Filter parameters for defect scans
const filteredParameters = computed(() => {
    const params = { ...props.details.parameters };
    if (!props.details.isDefect && params["Classification Inference"]) {
        delete params["Classification Inference"];
    }
    return params;
});

// Filter metrics for defect scans, removing the confidence level to be displayed separately
const filteredMetrics = computed(() => {
    const metrics = { ...props.details.scanMetrics };
    if (!props.details.isDefect && metrics["Classification Avg Confidence"]) {
        delete metrics["Classification Avg Confidence"];
    }
    // Remove confidence level so it's not rendered with a progress bar
    delete metrics["Anomaly Confidence Level"];
    return metrics;
});

// New computed property to style the anomaly confidence level text as a badge
const anomalyConfidenceStyle = computed(() => {
    const level = props.details.scanMetrics["Anomaly Confidence Level"];

    // Handle cases where level might be missing or not a string
    if (typeof level !== "string") {
        return { text: "N/A", class: "text-gray-500" };
    }

    const baseClass = "px-2 py-1 rounded-sm font-medium text-xs";

    // Convert to lowercase for case-insensitive matching
    switch (level.toLowerCase()) {
        case "high":
            return {
                text: "High",
                class: `${baseClass} bg-red-100 text-red-800 dark:bg-red-800/30 dark:text-red-400`,
            };
        case "medium":
            return {
                text: "Medium",
                class: `${baseClass} bg-yellow-100 text-yellow-800 dark:bg-yellow-800/30 dark:text-yellow-400`,
            };
        case "low":
            return {
                text: "Low",
                class: `${baseClass} bg-green-100 text-green-800 dark:bg-green-800/30 dark:text-green-400`,
            };
        default:
            return { text: "N/A", class: "text-gray-500" };
    }
});

// --- Helper Functions ---

/**
 * Formats the metric value for display.
 * Converts string numbers to numbers, then formats to 3 decimal places.
 */
const formatMetricValue = (key, value) => {
    // Attempt to parse the value as a float. This handles both numbers and string-numbers.
    const num = parseFloat(value);

    // Check if the result is a valid number (not NaN).
    // This will be true for numbers (e.g., 0.862) and numeric strings ("0.862"),
    // but false for non-numeric strings (e.g., "Low").
    if (!isNaN(num)) {
        return num.toFixed(3);
    }

    // If it's not a number, return the original value.
    return value;
};

/**
 * Determines the display color of the progress bar based on the metric's meaning.
 */
const getMetricColor = (key, value) => {
    switch (key) {
        case "Anomaly Score":
            // Higher score is more critical
            if (value > 0.8) return "bg-red-500";
            if (value > 0.6) return "bg-yellow-500";
            return "bg-green-500";

        case "Classification Avg Confidence":
            // For defects, higher confidence is better (more certain)
            if (value > 0.8) return "bg-green-500";
            if (value > 0.5) return "bg-yellow-500";
            return "bg-red-500";

        case "Anomaly Threshold":
            // Use a neutral color for a threshold setting
            return "bg-blue-500";

        default:
            // Fallback for any other metrics
            return "bg-gray-400";
    }
};

/**
 * Calculates the width of the progress bar as a percentage.
 */
const getMetricPercentage = (key, value) => {
    switch (key) {
        case "Anomaly Score":
        case "Anomaly Threshold":
        case "Classification Avg Confidence":
            // Convert decimal (0 to 1) to a percentage
            return (value || 0) * 100;

        default:
            // Fallback for any other numeric metrics
            return typeof value === "number" ? value * 100 : 0;
    }
};
</script>

<template>
    <div class="space-y-6">
        <!-- Main Technical Details -->
        <div
            class="bg-white dark:bg-dark-800 shadow-sm dark:shadow-none px-6 py-5 border border-gray-200 dark:border-none rounded-lg"
        >
            <h3
                class="flex items-center gap-4 mb-6 font-semibold text-gray-900 dark:text-dark-50 text-lg"
            >
                <font-awesome-icon icon="fa-solid fa-code" />
                Technical Analysis Details
            </h3>

            <div class="gap-8 grid grid-cols-1 lg:grid-cols-2">
                <!-- Left Side: Parameters and Metrics -->
                <div class="space-y-6">
                    <!-- Analysis Parameters -->
                    <div>
                        <h2
                            class="mb-3 font-medium text-gray-800 dark:text-dark-100 text-sm-plus truncate tracking-wide"
                        >
                            Analysis Parameters
                        </h2>
                        <dl
                            class="border-gray-200 dark:border-dark-600 border-t text-sm"
                        >
                            <div
                                v-for="(value, key) in filteredParameters"
                                :key="key"
                                class="grid grid-cols-2 px-0 py-2 border-gray-200 dark:border-dark-600 border-b"
                            >
                                <dt class="font-mono truncate">{{ key }}</dt>
                                <dd class="font-mono text-right truncate">
                                    {{ value }}
                                </dd>
                            </div>
                        </dl>
                    </div>

                    <!-- Scan Analysis Metrics -->
                    <div>
                        <h2
                            class="mb-3 font-medium text-gray-800 dark:text-dark-100 text-sm-plus truncate tracking-wide"
                        >
                            Scan Analysis Metrics
                        </h2>
                        <div class="space-y-4">
                            <div class="flex justify-between mb-2 text-sm">
                                <span class="truncate"
                                    >Anomaly Confidence Level</span
                                >
                                <span :class="anomalyConfidenceStyle.class">
                                    {{ anomalyConfidenceStyle.text }}
                                </span>
                            </div>
                            <div
                                v-for="(value, key) in filteredMetrics"
                                :key="key"
                            >
                                <div class="flex justify-between mb-2 text-sm">
                                    <span class="truncate">{{ key }}</span>
                                    <span class="truncate">{{
                                        formatMetricValue(key, value)
                                    }}</span>
                                </div>
                                <div
                                    class="bg-gray-200 dark:bg-dark-700 rounded-full w-full h-2.5"
                                >
                                    <div
                                        class="rounded-full h-2.5"
                                        :class="getMetricColor(key, value)"
                                        :style="{
                                            width: `${getMetricPercentage(
                                                key,
                                                value
                                            )}%`,
                                        }"
                                    ></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right Side: Raw JSON Data -->
                <div class="flex flex-col h-full">
                    <h2
                        class="mb-3 font-medium text-gray-800 dark:text-dark-100 text-sm-plus truncate tracking-wide"
                    >
                        Raw Analysis Data
                    </h2>
                    <div
                        class="flex-1 border border-gray-200 dark:border-dark-600 rounded-lg"
                    >
                        <pre
                            class="p-4 text-gray-800 dark:text-dark-100 text-xs whitespace-pre-wrap"
                        ><code>{{ formattedRawData }}</code></pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- Security Threat Analysis (Only show if threat data exists) -->
        <div
            v-if="details.hasThreat"
            class="bg-white dark:bg-dark-800 shadow-sm dark:shadow-none px-6 py-5 border border-gray-200 dark:border-none rounded-lg"
        >
            <h3
                class="flex items-center gap-4 mb-6 font-semibold text-gray-900 dark:text-dark-50 text-lg"
            >
                <font-awesome-icon icon="fa-solid fa-shield" />
                Security Threat Analysis
            </h3>

            <div class="gap-6 grid grid-cols-1 lg:grid-cols-2">
                <!-- Threat Summary -->
                <div class="space-y-4">
                    <div
                        class="bg-yellow-50 dark:bg-yellow-900/20 p-4 border border-yellow-200 dark:border-yellow-800 rounded-lg"
                    >
                        <div class="gap-4 grid grid-cols-2 text-sm">
                            <div>
                                <span
                                    class="font-medium text-gray-700 dark:text-dark-200"
                                    >Status:</span
                                >
                                <div class="mt-1">
                                    <span
                                        :class="{
                                            'bg-yellow-100 text-yellow-800 dark:bg-yellow-800/30 dark:text-yellow-400':
                                                details.threatData?.status ===
                                                'SUSPICIOUS',
                                            'bg-red-100 text-red-800 dark:bg-red-800/30 dark:text-red-400':
                                                details.threatData?.status ===
                                                'MALICIOUS',
                                            'bg-green-100 text-green-800 dark:bg-green-800/30 dark:text-green-400':
                                                details.threatData?.status ===
                                                'CLEAN',
                                        }"
                                        class="px-2 py-1 rounded-full font-medium text-xs"
                                    >
                                        {{
                                            details.threatData?.status ||
                                            "Unknown"
                                        }}
                                    </span>
                                </div>
                            </div>
                            <div>
                                <span
                                    class="font-medium text-gray-700 dark:text-dark-200"
                                    >Risk Level:</span
                                >
                                <div class="mt-1">
                                    <span
                                        :class="{
                                            'bg-yellow-100 text-yellow-800 dark:bg-yellow-800/30 dark:text-yellow-400':
                                                details.threatData
                                                    ?.riskLevel === 'MEDIUM',
                                            'bg-red-100 text-red-800 dark:bg-red-800/30 dark:text-red-400':
                                                details.threatData
                                                    ?.riskLevel === 'HIGH',
                                            'bg-green-100 text-green-800 dark:bg-green-800/30 dark:text-green-400':
                                                details.threatData
                                                    ?.riskLevel === 'LOW',
                                        }"
                                        class="px-2 py-1 rounded-full font-medium text-xs"
                                    >
                                        {{
                                            details.threatData?.riskLevel ||
                                            "Unknown"
                                        }}
                                    </span>
                                </div>
                            </div>
                            <div class="col-span-2">
                                <span
                                    class="font-medium text-gray-700 dark:text-dark-200"
                                    >File Hash:</span
                                >
                                <div
                                    class="mt-1 font-mono text-gray-600 dark:text-dark-300 text-xs break-all"
                                >
                                    {{ details.threatData?.fileHash || "N/A" }}
                                </div>
                            </div>
                            <div>
                                <span
                                    class="font-medium text-gray-700 dark:text-dark-200"
                                    >Scan Time:</span
                                >
                                <div
                                    class="mt-1 text-gray-600 dark:text-dark-300"
                                >
                                    {{ details.threatData?.scanTime || "N/A" }}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Security Flags -->
                    <div v-if="details.threatData?.securityFlags?.length">
                        <h4
                            class="mb-2 font-medium text-gray-800 dark:text-dark-100"
                        >
                            Security Flags Detected:
                        </h4>
                        <div class="flex flex-wrap gap-2">
                            <span
                                v-for="flag in details.threatData.securityFlags"
                                :key="flag"
                                class="bg-red-100 dark:bg-red-800/30 px-2 py-1 rounded-md text-red-800 dark:text-red-400 text-xs"
                            >
                                {{ flag }}
                            </span>
                        </div>
                    </div>

                    <!-- Detailed Analysis -->
                    <div v-if="details.threatData?.detailedAnalysis">
                        <h4
                            class="mb-2 font-medium text-gray-800 dark:text-dark-100"
                        >
                            Detailed Analysis:
                        </h4>
                        <div class="space-y-2">
                            <div
                                v-for="(analysis, key) in details.threatData
                                    .detailedAnalysis"
                                :key="key"
                                class="bg-gray-50 dark:bg-dark-700 p-3 rounded-md"
                            >
                                <div
                                    class="font-medium text-gray-700 dark:text-dark-200 text-sm"
                                >
                                    {{ key }}:
                                </div>
                                <div
                                    class="text-gray-600 dark:text-dark-300 text-sm"
                                >
                                    {{ analysis }}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Possible Attack Types -->
                <div v-if="details.threatData?.possibleAttacks?.length">
                    <h4
                        class="mb-3 font-medium text-gray-800 dark:text-dark-100"
                    >
                        Possible Attack Types:
                    </h4>
                    <div class="space-y-3 max-h-96 overflow-y-auto">
                        <div
                            v-for="(attack, index) in details.threatData
                                .possibleAttacks"
                            :key="index"
                            class="bg-red-50 dark:bg-red-900/20 p-3 border border-red-200 dark:border-red-800 rounded-md"
                        >
                            <p class="text-red-800 dark:text-red-400 text-sm">
                                {{ attack }}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
