<script setup>
import { ref, computed, reactive, onMounted, watch } from "vue";
import { useForm } from "@inertiajs/vue3";
import { router } from "@inertiajs/vue3";

// --- Import Components & Utils ---
import AccountSettings from "./Components/AccountSettings.vue";
import DetectionSettings from "./Components/DetectionSettings.vue";
import AdvancedSettings from "./Components/AdvancedSettings.vue";
import ConfirmationModal from "./Components/Modals/ConfirmationModal.vue";
import { successToast } from "@/utils/swal";

// --- Props ---
// NEW: Define the props that will be passed from the Laravel backend.
const props = defineProps({
    savedSettings: {
        type: Object,
        required: true,
    },
});

// --- State ---
const activeTab = ref("account");
const pristineSettings = ref("");
const isDangerZoneResetModalVisible = ref(false);
const isClearDataModalVisible = ref(false);

// --- Default Settings Data (for resetting) ---
const defaultSettings = {
    detection: {
        anomalyThreshold: 0.75,
        defectThreshold: 0.85,
        exportFormat: "pdf",
    },
};

// --- Live Settings Data ---
// MODIFIED: Initialize the settings with the data passed from the server.
const settings = reactive({
    detection: JSON.parse(JSON.stringify(props.savedSettings.detection)),
    advanced: {
        systemInfo: {
            "System Version": "v2.0.0",
            "API Status": "Connected",
            "Models Loaded": "Yes",
            Database: "Connected",
        },
    },
});

// --- Form Management ---
const settingsForm = useForm({
    detection: settings.detection,
});

// --- Computed Properties ---
const hasUnsavedChanges = computed(() => {
    if (!pristineSettings.value) return false;
    const pristine = JSON.parse(pristineSettings.value);
    return (
        JSON.stringify(settings.detection) !==
        JSON.stringify(pristine.detection)
    );
});

const updatePristineState = () => {
    pristineSettings.value = JSON.stringify({
        detection: settings.detection,
    });
};

// --- Watchers ---
watch(
    settings,
    (newSettings) => {
        settingsForm.detection = newSettings.detection;
    },
    { deep: true }
);

// --- Lifecycle Hooks ---
// MODIFIED: This now simply takes a snapshot of the initial server-provided state.
onMounted(() => {
    updatePristineState();
});

// --- Tab Configuration ---
const tabs = [
    {
        id: "account",
        name: "Account",
        component: AccountSettings,
        icon: "fa-solid fa-user",
    },
    {
        id: "detection",
        name: "Detection Settings",
        component: DetectionSettings,
        icon: "fa-solid fa-magnifying-glass",
    },
    {
        id: "advanced",
        name: "Advanced",
        component: AdvancedSettings,
        icon: "fa-solid fa-gear",
    },
];

const activeTabComponent = computed(() => {
    return tabs.find((tab) => tab.id === activeTab.value)?.component;
});

const activeTabProps = computed(() => {
    switch (activeTab.value) {
        case "account":
            return {};
        case "detection":
            return {
                settings: settings.detection,
            };
        case "advanced":
            return {
                settings: settings.advanced,
            };
        default:
            return {};
    }
});

// --- Event Handlers ---
const saveSettings = (showToast = true) => {
    settingsForm.patch(route("settings.detection_settings.update"), {
        onSuccess: () => {
            if (showToast) {
                successToast("Settings saved successfully!");
            }
            updatePristineState();
        },
        onError: (errors) => {
            console.error("Failed to save settings:", errors);
        },
    });
};

const resetUnsavedChanges = () => {
    const lastSavedState = JSON.parse(pristineSettings.value);
    settings.detection = lastSavedState.detection;
    successToast("Changes have been discarded.");
};

// --- "Danger Zone" Event Handlers ---
const handleClearData = () => {
    isClearDataModalVisible.value = true;
};

const handleConfirmClearData = () => {
    router.delete(route("settings.clear_all_data"), {
        onSuccess: () => {
            // The success message will come from the backend redirect
            isClearDataModalVisible.value = false;
            successToast("All analysis data has been cleared.");
        },
        onError: (errors) => {
            console.error("Failed to clear data:", errors);
            // Optionally show an error toast here
        },
    });
};

const handleResetSettings = () => {
    isDangerZoneResetModalVisible.value = true;
};

const handleConfirmDangerZoneReset = () => {
    settings.detection = JSON.parse(JSON.stringify(defaultSettings.detection));
    isDangerZoneResetModalVisible.value = false;
    successToast("Settings have been reset to their default values.");
    saveSettings(false);
};
</script>

<template>
    <div>
        <!-- Tab Navigation -->
        <div class="mb-6">
            <div
                class="flex md:flex-row flex-col md:justify-between md:items-end gap-4 md:border-gray-200 md:dark:border-dark-700 md:border-b"
            >
                <!-- Mobile version -- Dropdown -->
                <div class="md:hidden">
                    <label for="tabs" class="sr-only">Select a tab</label>
                    <div class="input-root undefined">
                        <div class="relative input-wrapper">
                            <select
                                id="tabs"
                                v-model="activeTab"
                                class="peer block mt-1 ltr:pr-9 rtl:pl-9 border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:hover:border-dark-400 dark:focus:border-primary-500 dark:border-dark-450 w-full text-sm form-select-base drop-down-tab form-select"
                            >
                                <option
                                    v-for="tab in tabs"
                                    :key="tab.id"
                                    :value="tab.id"
                                >
                                    {{ tab.name }}
                                </option>
                            </select>
                            <div
                                class="top-0 ltr:right-0 rtl:left-0 absolute flex justify-center items-center w-9 h-full text-gray-400 dark:text-dark-300 dark:peer-focus:text-primary-500 peer-focus:text-primary-600 transition-colors pointer-events-none suffix"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 20 20"
                                    fill="currentColor"
                                    aria-hidden="true"
                                    data-slot="icon"
                                    class="w-2/3"
                                >
                                    <path
                                        fill-rule="evenodd"
                                        d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                                        clip-rule="evenodd"
                                    ></path>
                                </svg>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Desktop version -- Tabs -->
                <div class="hidden md:block">
                    <nav class="flex space-x-6 -mb-px" aria-label="Tabs">
                        <button
                            v-for="tab in tabs"
                            :key="tab.id"
                            @click="activeTab = tab.id"
                            class="gap-2 btn-base shrink-0"
                            :class="[
                                activeTab === tab.id
                                    ? 'whitespace-nowrap border-b-2 px-3 py-2 font-medium  border-primary-600 text-primary-600 dark:border-primary-500 dark:text-primary-400'
                                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-dark-300 dark:hover:text-dark-100 dark:hover:border-dark-500',
                                'whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm',
                            ]"
                        >
                            <font-awesome-icon :icon="tab.icon" />
                            {{ tab.name }}
                        </button>
                    </nav>
                </div>

                <!-- Action buttons for unsaved changes -->
                <div
                    v-if="hasUnsavedChanges"
                    class="max-md:hidden flex items-center self-end md:self-auto gap-3 md:pb-2"
                >
                    <button
                        @click="resetUnsavedChanges"
                        class="gap-2 bg-gray-150 hover:bg-gray-200 focus:bg-gray-200 active:bg-gray-200/80 dark:active:bg-surface-1/90 dark:bg-surface-2 dark:hover:bg-surface-1 dark:focus:bg-surface-1 text-gray-900 dark:text-dark-50 btn-base btn"
                    >
                        <font-awesome-icon
                            icon="fa-solid fa-arrow-rotate-left"
                        />
                        Reset Changes
                    </button>
                    <button
                        @click="saveSettings"
                        :disabled="isSaving"
                        class="gap-2 bg-this hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker max-w-lg text-white btn-base btn this:primary"
                    >
                        <span
                            v-if="isSaving"
                            class="flex justify-center items-center gap-2"
                        >
                            <div
                                class="border-2 border-white border-t-transparent rounded-full w-4 h-4 animate-spin"
                            ></div>
                            Saving...
                        </span>
                        <span v-else class="flex items-center gap-2 truncate">
                            <font-awesome-icon icon="fa-solid fa-floppy-disk" />
                            Save Settings
                        </span>
                    </button>
                </div>
            </div>
        </div>

        <!-- Dynamic Tab Content -->
        <div>
            <keep-alive>
                <component
                    :is="activeTabComponent"
                    v-bind="activeTabProps"
                    @clear-data="handleClearData"
                    @reset-settings="handleResetSettings"
                />
            </keep-alive>
        </div>

        <!-- Small Screen Buttons for unsaved changes -->
        <div
            v-if="hasUnsavedChanges"
            class="md:hidden flex justify-end items-center gap-3 mt-6 pt-6 border-gray-200 dark:border-dark-700 border-t"
        >
            <button
                @click="resetUnsavedChanges"
                class="gap-2 bg-gray-150 hover:bg-gray-200 focus:bg-gray-200 active:bg-gray-200/80 dark:active:bg-surface-1/90 dark:bg-surface-2 dark:hover:bg-surface-1 dark:focus:bg-surface-1 text-gray-900 dark:text-dark-50 btn-base btn"
            >
                <font-awesome-icon icon="fa-solid fa-arrow-rotate-left" />
                Reset Changes
            </button>
            <button
                @click="saveSettings"
                :disabled="isSaving"
                class="gap-2 bg-this hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker w-36 text-white btn-base btn this:primary"
            >
                <span
                    v-if="isSaving"
                    class="flex justify-center items-center gap-2"
                >
                    <div
                        class="border-2 border-white border-t-transparent rounded-full w-4 h-4 animate-spin"
                    ></div>
                    Saving...
                </span>
                <span v-else class="flex items-center gap-2">
                    <font-awesome-icon icon="fa-solid fa-floppy-disk" />
                    Save Settings
                </span>
            </button>
        </div>

        <!-- Modal for the DANGER ZONE "Reset All Settings" -->
        <ConfirmationModal
            :show="isDangerZoneResetModalVisible"
            title="Reset All Settings to Default?"
            message="Are you sure you want to reset all settings to their factory defaults? This action cannot be undone."
            confirm-text="Yes, Reset All Settings"
            variant="error"
            icon="fa-solid fa-triangle-exclamation"
            @close="isDangerZoneResetModalVisible = false"
            @confirm="handleConfirmDangerZoneReset"
        />

        <ConfirmationModal
            :show="isClearDataModalVisible"
            title="Clear All Analysis Data?"
            message="Are you sure you want to permanently delete all scan and session history? This action cannot be undone."
            confirm-text="Yes, Clear All Data"
            variant="error"
            icon="fa-solid fa-trash"
            @close="isClearDataModalVisible = false"
            @confirm="handleConfirmClearData"
        />
    </div>
</template>

<style scoped>
.dark .drop-down-tab {
    --bg-color: rgb(21, 22, 26);
}

.drop-down-tab {
    --bg-color: rgb(255, 255, 255);
}
</style>
