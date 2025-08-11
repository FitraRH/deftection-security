<script setup>
import { computed, onMounted } from "vue";

// --- Props ---
const props = defineProps({
    settings: {
        type: Object,
        required: true,
    },
});

// --- Emits ---
// Defines events for the dangerous actions. The parent component will handle these.
const emit = defineEmits(["update:settings", "clear-data", "reset-settings"]);

// Helper to get status text color
const getStatusColor = (status) => {
    if (status === "Connected" || status === "Yes") return "text-green-500";
    if (status === "Unknown") return "text-yellow-500";
    return "text-red-500";
};
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <h2
            class="mb-1 flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
        >
            Advanced Configuration
        </h2>
        <p class="mb-6 text-xs text-gray-400 dark:text-dark-300">
            Advanced settings for system behavior and performance.
        </p>

        <!-- Section 1: System Information -->
        <div class="p-4 border border-gray-200 rounded-lg dark:border-dark-600">
            <h4 class="font-medium text-gray-900 dark:text-dark-50">
                System Information
            </h4>
            <dl class="mt-2 text-sm">
                <div
                    v-for="(value, key) in props.settings.systemInfo"
                    :key="key"
                    class="grid grid-cols-2 py-1"
                >
                    <dt class="text-gray-400 dark:text-dark-300">{{ key }}</dt>
                    <dd
                        class="font-medium text-right"
                        :class="getStatusColor(value)"
                    >
                        {{ value }}
                    </dd>
                </div>
            </dl>
        </div>

        <hr class="my-6 border-gray-200 dark:border-dark-600" />

        <!-- Section 2: Danger Zone -->
        <div
            class="p-4 mt-6 border-2 border-dashed rounded-lg border-red-300 dark:border-red-500/50"
        >
            <h2
                class="flex items-center font-bold text-red-700 dark:text-red-400"
            >
                Danger Zone
            </h2>
            <div class="mt-4 space-y-4">
                <div
                    class="flex flex-col sm:flex-row sm:items-center sm:justify-between"
                >
                    <div>
                        <p
                            class="mb-1 text-sm font-medium text-gray-800 dark:text-dark-100"
                        >
                            Clear All Analysis Data
                        </p>
                        <p class="text-xs text-gray-400 dark:text-dark-300">
                            Permanently delete all analysis history and uploaded
                            images.
                        </p>
                    </div>
                    <button
                        @click="emit('clear-data')"
                        class="btn btn-base px-6 py-2 mt-2 gap-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md shadow-sm sm:mt-0 sm:ml-4 hover:bg-red-700"
                    >
                        <font-awesome-icon icon="fa-solid fa-trash-can" />
                        Clear Data
                    </button>
                </div>
                <div
                    class="flex flex-col pt-4 mt-4 border-t border-red-200 sm:flex-row sm:items-center sm:justify-between dark:border-red-500/30"
                >
                    <div>
                        <p
                            class="mb-1 text-sm font-medium text-gray-800 dark:text-dark-100"
                        >
                            Reset All Settings
                        </p>
                        <p class="text-xs text-gray-400 dark:text-dark-300">
                            Reset all configuration to factory defaults.
                        </p>
                    </div>
                    <button
                        @click="emit('reset-settings')"
                        class="btn btn-base px-7 py-2 mt-2 gap-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md shadow-sm sm:mt-0 sm:ml-4 hover:bg-red-700"
                    >
                        <font-awesome-icon
                            icon="fa-solid fa-clock-rotate-left"
                        />
                        Reset All
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.dark #maxConcurrent {
    --bg-color: rgb(21, 22, 26);
}

#maxConcurrent {
    --bg-color: rgb(255, 255, 255);
}
</style>
