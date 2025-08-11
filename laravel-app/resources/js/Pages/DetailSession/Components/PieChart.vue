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
const pieChartSeries = computed(() => {
    return [props.session.good_scans, props.session.defected_scans];
});

const pieChartOptions = computed(() => ({
    chart: { type: "pie" },
    labels: ["Good Scans", "Defected Scans"],
    colors: ["#22C55E", "#EF4444"], // Green, Red
    legend: {
        position: "bottom",
        labels: {
            colors:
                localStorage.getItem("theme") === "dark"
                    ? "#D1D5DB"
                    : "#374151",
        },
    },
    tooltip: {
        theme: localStorage.getItem("theme") === "dark" ? "dark" : "light",
    },
    dataLabels: {
        enabled: true,
        formatter: (val) => `${val.toFixed(1)}%`,
    },
}));
</script>

<template>
    <div class="p-6 bg-white rounded-lg shadow dark:bg-dark-800">
        <h2
            class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
        >
            Good vs. Defect Ratio
        </h2>
        <div class="h-auto mt-4">
            <VueApexCharts
                type="pie"
                :options="pieChartOptions"
                :series="pieChartSeries"
                height="450"
            />
        </div>
    </div>
</template>
