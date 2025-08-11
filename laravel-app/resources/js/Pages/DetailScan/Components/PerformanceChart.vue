<script setup>
import { computed } from "vue";
import VueApexCharts from "vue3-apexcharts";

// --- Props ---
const props = defineProps({
    performanceData: {
        type: Object,
        required: true,
        default: () => ({
            timeBreakdown: {
                preprocessing: 0.483,
                anomalyInference: 0.902,
                classificationInference: 0.119, // Only show if defect
                postprocessing: 0.918,
            },
            isDefect: false, // To determine if we show classification
        }),
    },
});

// --- ApexCharts Configuration ---

// --- Bar Chart for Processing Performance ---
const barChartSeries = computed(() => {
    const data = props.performanceData.timeBreakdown;
    const times = [
        data.preprocessing * 1000, // Convert to ms
        data.anomalyInference * 1000,
        data.postprocessing * 1000,
    ];

    if (props.performanceData.isDefect) {
        times.splice(2, 0, data.classificationInference * 1000); // Insert classification at index 2
    }

    return [
        {
            name: "Time (ms)",
            data: times,
        },
    ];
});

const barChartOptions = computed(() => {
    const categories = props.performanceData.isDefect
        ? [
              "Preprocessing",
              "Anomaly Inference",
              "Classification Inference",
              "Postprocessing",
          ]
        : ["Preprocessing", "Anomaly Inference", "Postprocessing"];

    return {
        chart: {
            type: "bar",
            toolbar: { show: false },
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: "55%",
                borderRadius: 4,
            },
        },
        dataLabels: {
            enabled: false,
        },
        stroke: {
            show: true,
            width: 2,
            colors: ["transparent"],
        },
        xaxis: {
            categories: categories,
            labels: {
                style: {
                    colors: "#D1D5DB",
                },
            },
        },
        yaxis: {
            title: {
                text: "Time (ms)",
                style: {
                    color: "#D1D5DB",
                },
            },
            labels: {
                style: {
                    colors: "#9CA3AF",
                },
                formatter: (val) => val.toFixed(1),
            },
        },
        fill: {
            opacity: 1,
            colors: props.performanceData.isDefect
                ? ["#3B82F6", "#10B981", "#EF4444", "#F59E0B"]
                : ["#3B82F6", "#10B981", "#F59E0B"],
        },
        tooltip: {
            theme: "dark",
            y: {
                formatter: (val) => `${val.toFixed(1)}ms`,
            },
        },
        grid: {
            borderColor: "#4B5563",
        },
    };
});

// Calculate percentages for display
const percentages = computed(() => {
    const data = props.performanceData.timeBreakdown;
    const total =
        data.preprocessing +
        data.anomalyInference +
        data.postprocessing +
        (props.performanceData.isDefect ? data.classificationInference : 0);

    const result = {
        preprocessing: ((data.preprocessing / total) * 100).toFixed(1),
        anomalyInference: ((data.anomalyInference / total) * 100).toFixed(1),
        postprocessing: ((data.postprocessing / total) * 100).toFixed(1),
    };

    if (props.performanceData.isDefect) {
        result.classificationInference = (
            (data.classificationInference / total) *
            100
        ).toFixed(1);
    }

    return result;
});
</script>

<template>
    <div
        class="bg-white dark:bg-dark-800 shadow-sm dark:shadow-none px-6 py-5 border border-gray-200 dark:border-none rounded-lg"
    >
        <h3
            class="flex items-center gap-4 mb-6 font-semibold text-gray-900 dark:text-dark-50 text-lg"
        >
            <font-awesome-icon icon="fa-solid fa-chart-pie" />
            Processing Performance Analysis
        </h3>

        <div class="gap-8 grid grid-cols-1 lg:grid-cols-2">
            <!-- Left Side: Pie Chart -->
            <div
                class="p-4 border border-gray-200 dark:border-dark-600 rounded-lg"
            >
                <h5
                    class="mb-4 font-medium text-gray-700 dark:text-dark-200 text-center"
                >
                    Processing Time Table
                </h5>
                <table
                    class="divide-y divide-gray-200 dark:divide-dark-600 min-w-full"
                >
                    <thead class="bg-gray-50 dark:bg-dark-700">
                        <tr>
                            <th
                                class="px-6 py-3 font-medium text-gray-500 dark:text-dark-300 text-xs text-left uppercase tracking-wider"
                            >
                                Processing Stage
                            </th>
                            <th
                                class="px-6 py-3 font-medium text-gray-500 dark:text-dark-300 text-xs text-left uppercase tracking-wider"
                            >
                                Time (ms)
                            </th>
                            <th
                                class="px-6 py-3 font-medium text-gray-500 dark:text-dark-300 text-xs text-left uppercase tracking-wider"
                            >
                                Percentage
                            </th>
                        </tr>
                    </thead>
                    <tbody
                        class="bg-white dark:bg-dark-800 divide-y divide-gray-200 dark:divide-dark-600"
                    >
                        <tr>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                Preprocessing
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{
                                    (
                                        performanceData.timeBreakdown
                                            .preprocessing * 1000
                                    ).toFixed(1)
                                }}
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{ percentages.preprocessing }}%
                            </td>
                        </tr>
                        <tr>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                Anomaly Inference
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{
                                    (
                                        performanceData.timeBreakdown
                                            .anomalyInference * 1000
                                    ).toFixed(1)
                                }}
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{ percentages.anomalyInference }}%
                            </td>
                        </tr>
                        <tr v-if="performanceData.isDefect">
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                Classification Inference
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{
                                    (
                                        performanceData.timeBreakdown
                                            .classificationInference * 1000
                                    ).toFixed(1)
                                }}
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{ percentages.classificationInference }}%
                            </td>
                        </tr>
                        <tr>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                Postprocessing
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{
                                    (
                                        performanceData.timeBreakdown
                                            .postprocessing * 1000
                                    ).toFixed(1)
                                }}
                            </td>
                            <td
                                class="px-6 py-4 text-gray-900 dark:text-dark-100 text-sm whitespace-nowrap"
                            >
                                {{ percentages.postprocessing }}%
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Right Side: Bar Chart -->
            <div
                class="p-4 border border-gray-200 dark:border-dark-600 rounded-lg"
            >
                <h5
                    class="mb-4 font-medium text-gray-700 dark:text-dark-200 text-center"
                >
                    Processing Time Distribution
                </h5>
                <VueApexCharts
                    type="bar"
                    height="350"
                    :options="barChartOptions"
                    :series="barChartSeries"
                />
            </div>
        </div>
    </div>
</template>
