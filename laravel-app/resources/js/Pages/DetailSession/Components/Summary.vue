<script setup>
// --- Props ---
// This component receives the main session object to display its summary data.
const props = defineProps({
    session: {
        type: Object,
        required: true,
    },
});
</script>

<template>
    <!-- Section 2: Key Metrics Summary -->
    <div class="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div class="p-4 bg-white rounded-lg shadow dark:bg-dark-800 space-y-1">
            <p class="text-xs-plus uppercase">Duration</p>
            <p class="text-2xl font-semibold text-gray-800 dark:text-dark-100">
                {{ session.duration }}
            </p>
        </div>
        <div class="p-4 bg-white rounded-lg shadow dark:bg-dark-800 space-y-1">
            <p class="text-xs-plus uppercase">Total Scans</p>
            <p class="text-2xl font-bold text-gray-800 dark:text-dark-100">
                {{ session.total_scans }}
            </p>
        </div>
        <div class="p-4 bg-white rounded-lg shadow dark:bg-dark-800 space-y-1">
            <p class="text-xs-plus uppercase">Good / Defect</p>
            <p class="text-2xl font-bold">
                <span class="text-green-500">{{ session.good_scans }}</span>
                /
                <span class="text-red-500">{{ session.defected_scans }}</span>
            </p>
        </div>
        <div class="p-4 bg-white rounded-lg shadow dark:bg-dark-800 space-y-1">
            <p class="text-xs-plus uppercase">Defect Rate</p>
            <p class="text-2xl font-bold text-red-500">
                {{ session.defect_rate.toFixed(1) }}%
            </p>
        </div>
    </div>
</template>
