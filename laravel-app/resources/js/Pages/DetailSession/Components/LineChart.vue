<script setup>
import { computed } from "vue";
import VueApexCharts from "vue3-apexcharts";

// --- Props ---
const props = defineProps({
    session: {
        type: Object,
        required: true,
    },
});

// --- Chart Configuration ---
const lineChartSeries = computed(() => {
    // This is a simplified example. In a real app, you'd process timestamps from session.scans.
    const scanCounts = [5, 10, 8, 15, 12, 20, 18, 25, 22, 30]; // Mock data
    return [
        {
            name: "Scans Detected",
            data: scanCounts,
        },
    ];
});

const lineChartOptions = computed(() => ({
    chart: { type: "area", toolbar: { show: false }, zoom: { enabled: false } },
    colors: ["#3B82F6"], // Blue
    dataLabels: { enabled: false },
    stroke: { curve: "smooth", width: 2 },
    xaxis: {
        categories: [
            "0m",
            "5m",
            "10m",
            "15m",
            "20m",
            "25m",
            "30m",
            "35m",
            "40m",
            "45m",
        ],
        labels: {
            style: {
                colors:
                    localStorage.getItem("theme") === "dark"
                        ? "#9CA3AF"
                        : "#6B7280",
            },
        },
    },
    yaxis: {
        labels: {
            style: {
                colors:
                    localStorage.getItem("theme") === "dark"
                        ? "#9CA3AF"
                        : "#6B7280",
            },
        },
    },
    grid: {
        borderColor:
            localStorage.getItem("theme") === "dark" ? "#374151" : "#E5E7EB",
    },
    tooltip: {
        theme: localStorage.getItem("theme") === "dark" ? "dark" : "light",
    },
}));
</script>

<template>
    <div class="p-6 bg-white rounded-lg shadow dark:bg-dark-800">
        <h2
            class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
        >
            Scans Over Time
        </h2>
        <div class="h-auto mt-4">
            <VueApexCharts
                type="area"
                :options="lineChartOptions"
                :series="lineChartSeries"
            />
        </div>
    </div>
</template>
