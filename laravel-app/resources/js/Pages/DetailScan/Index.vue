<script setup>
import { ref, computed, onMounted } from "vue";

import AnalysisSummary from "./Components/AnalysisSummary.vue";
import ImageComparison from "./Components/ImageComparison.vue";
import DefectDetailCharts from "./Components/DefectDetailCharts.vue";
import PerformanceChart from "./Components/PerformanceChart.vue";
import ProductQuality from "./Components/ProductQuality.vue";
import TechnicalDetails from "./Components/TechnicalDetails.vue";

// --- Props Definition ---
const props = defineProps({
    analysis: {
        type: Object,
        required: true,
    },
    title: {
        type: String,
        default: "Scan Analysis Details",
    },
    scan: {
        type: Object,
        default: () => ({}),
    },
});

// --- Updated Mock Data with new structure ---
const mockDefectAnalysisData = {
    id: 27,
    status: "defect",
    summary: {
        imageName: "voluptatem.png",
        originalSize: "Unknown",
        analysisDate: "30/07/2025, 12:03:00",
        scannedBy: "user 1 (user@example.com)",
        finalDecision: "DEFECT",
        anomalyScore: 0.862,
        anomalyConfidenceLevel: "Low", // This is anomaly confidence level
        status: "Completed",
        // Processing times for performance card
        totalProcessingTime: "2.421s",
        preprocessingTime: "0.483s",
        anomalyInferenceTime: "0.902s",
        classificationInferenceTime: "0.119s",
        postprocessingTime: "0.918s",
    },
    visuals: {
        status: "defect",
        originalImageUrl:
            "https://placehold.co/600x400/FFFFFF/000000?text=Original+Image",
        analyzedImageUrl:
            "https://placehold.co/600x400/FF0000/FFFFFF?text=Analyzed+Image",
    },
    defects: [
        {
            name: "Discoloration",
            confidence: 9,
            coverage: 21.19,
            severity: "MEDIUM",
            location: { x: 84, y: 10, width: 50, height: 30 },
            explanation:
                "Color variation from the expected appearance, indicating potential quality issues.",
            regions: 1,
        },
        {
            name: "Missing Part",
            confidence: 7,
            coverage: 22.0,
            severity: "MEDIUM",
            location: { x: 34, y: 23, width: 40, height: 25 },
            explanation:
                "A component or part that should be present is absent from the expected location.",
            regions: 1,
        },
        {
            name: "Missing Part",
            confidence: 0,
            coverage: 32.64,
            severity: "MEDIUM",
            location: { x: 66, y: 31, width: 35, height: 40 },
            explanation:
                "A component or part that should be present is absent from the expected location.",
            regions: 1,
        },
    ],
    performance: {
        timeBreakdown: {
            preprocessing: 0.483,
            anomalyInference: 0.902,
            classificationInference: 0.119,
            postprocessing: 0.918,
        },
        isDefect: true,
    },
    technical: {
        parameters: {
            "Total Processing Time": "2.421s",
            "Image Preprocessing": "0.483s",
            "Anomaly Inference": "0.902s",
            "Classification Inference": "0.119s",
            Postprocessing: "0.918s",
            "Model Used": "HRNet + Anomalib",
        },
        scanMetrics: {
            "Anomaly Score": 0.862,
            "Anomaly Confidence Level": "Low",
            "Anomaly Threshold": 0.622,
            "Classification Avg Confidence": 0.053,
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
                "Ut commodi atque pariatur et. Est ad ipsum corrupti veniam doloribus sed. Officiis omnis dolorem sit sit facere aliquid.",
                "Quo voluptatem facere non voluptas voluptas dolore aut. Vitae optio error ut enim necessitatibus ab. Consequatur provident odio voluptatibus facilis dolores quia placeat.",
            ],
        },
        isDefect: true,
        hasThreat: true,
        rawData: {
            analysis_date: "2025-07-30 12:03:00",
            anomaly_score: 0.862,
            anomaly_confidence_level: "Low",
            anomaly_threshold: 0.622,
            final_decision: "DEFECT",
            id: 27,
            image_name: "voluptatem.png",
            defects: ["discoloration", "missing_part"],
            threat_status: "suspicious",
        },
    },
};

const mockGoodAnalysisData = {
    id: 16,
    status: "good",
    summary: {
        imageName: "good_sample.png",
        originalSize: "1024x768",
        analysisDate: "10/08/2025, 14:30:15",
        scannedBy: "user 2 (admin@example.com)",
        finalDecision: "GOOD",
        anomalyScore: 0.245,
        anomalyConfidenceLevel: "Low",
        status: "Completed",
        totalProcessingTime: "1.234s",
        preprocessingTime: "0.312s",
        anomalyInferenceTime: "0.756s",
        classificationInferenceTime: "0.000s", // No classification for good scans
        postprocessingTime: "0.166s",
    },
    visuals: {
        status: "good",
        originalImageUrl:
            "https://placehold.co/600x400/00FF00/000000?text=Good+Image",
        analyzedImageUrl: "",
    },
    defects: [],
    performance: {
        timeBreakdown: {
            preprocessing: 0.312,
            anomalyInference: 0.756,
            classificationInference: 0.0,
            postprocessing: 0.166,
        },
        isDefect: false,
    },
    technical: {
        parameters: {
            "Total Processing Time": "1.234s",
            "Image Preprocessing": "0.312s",
            "Anomaly Inference": "0.756s",
            Postprocessing: "0.166s",
            "Model Used": "HRNet + Anomalib",
        },
        scanMetrics: {
            "Anomaly Score": 0.245,
            "Anomaly Confidence Level": "Low",
            "Anomaly Threshold": 0.5,
        },
        threatData: null,
        isDefect: false,
        hasThreat: false,
        rawData: {
            analysis_date: "2025-08-10 14:30:15",
            anomaly_score: 0.245,
            anomaly_confidence_level: "Low",
            anomaly_threshold: 0.5,
            final_decision: "GOOD",
            id: 16,
            image_name: "good_sample.png",
            defects: [],
            threat_status: null,
        },
    },
};

// --- Simple check and console output ---
const analysis = ref(null);

onMounted(() => {
    if (props.analysis && props.analysis.id) {
        analysis.value = props.analysis;
    } else {
        console.log(" Using MOCK data - database failed:", props.analysis);
        // Use defect mock by default for testing, you can change this logic
        analysis.value = mockDefectAnalysisData;
    }
});

// --- Computed Properties ---
const isDefectStatus = computed(() => analysis.value?.status === "defect");
const isGoodStatus = computed(() => analysis.value?.status === "good");
const hasValidAnalysis = computed(
    () => analysis.value && analysis.value.summary
);

const handleDownload = () => {
    console.log("Download clicked for analysis ID:", analysis.value.id);
};

const handleDelete = () => {
    console.log("Delete clicked for analysis ID:", analysis.value.id);
};
</script>

<template>
    <!-- Error State -->
    <div
        v-if="!hasValidAnalysis"
        class="bg-red-50 p-6 border border-red-200 rounded-lg"
    >
        <div class="flex items-center">
            <div class="flex-shrink-0">
                <svg
                    class="w-5 h-5 text-red-400"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                >
                    <path
                        fill-rule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clip-rule="evenodd"
                    />
                </svg>
            </div>
            <div class="ml-3">
                <h3 class="font-medium text-red-800 text-sm">
                    Analysis Data Missing
                </h3>
                <div class="mt-2 text-red-700 text-sm">
                    <p>
                        Unable to load scan analysis data. Please try refreshing
                        the page or contact support.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div v-else class="flex flex-col gap-6">
        <!-- Page Title -->
        <div class="flex justify-between pb-4 border-gray-200 border-b">
            <div class="flex flex-col gap-1">
                <h1 class="font-bold text-gray-900 dark:text-dark-50 text-2xl">
                    {{ title }}
                </h1>
                <p class="mt-1 text-sm">
                    Analysis for {{ analysis.summary?.imageName }} • Scan ID:
                    {{ analysis.id }} •
                    {{ analysis.summary?.analysisDate }}
                </p>
            </div>
            <div class="flex items-end gap-3">
                <button
                    @click="handleDownload"
                    class="gap-2 bg-this hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker text-white btn-base btn this:primary"
                >
                    <font-awesome-icon icon="fa-solid fa-download" />
                    <span>Download</span>
                </button>
                <button
                    @click="handleDelete"
                    class="gap-2 bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 text-white transition-colors btn-base btn"
                >
                    <font-awesome-icon icon="fa-solid fa-trash-can" />
                    <span>Delete</span>
                </button>
            </div>
        </div>

        <!-- Analysis Summary (Updated with new structure) -->
        <AnalysisSummary :summary="analysis.summary" />

        <!-- Image Comparison -->
        <ImageComparison :comparison-data="analysis.visuals" />

        <!-- Defect Details (only for defect status) -->
        <DefectDetailCharts v-if="isDefectStatus" :defects="analysis.defects" />

        <!-- Performance Chart (Updated - now shows processing performance) -->
        <PerformanceChart :performance-data="analysis.performance" />

        <!-- Product Quality (only for good status) -->
        <ProductQuality v-if="isGoodStatus" />

        <!-- Technical Details (Updated with new structure) -->
        <TechnicalDetails :details="analysis.technical" />
    </div>
</template>
