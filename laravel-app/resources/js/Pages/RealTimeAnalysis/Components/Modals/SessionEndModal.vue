<script setup>
import {
    Dialog,
    DialogPanel,
    DialogTitle,
    TransitionRoot,
    TransitionChild,
} from "@headlessui/vue";
import { computed } from "vue";

// Defines the component's props for visibility, state, and session statistics.
const props = defineProps({
    isVisible: {
        type: Boolean,
        default: false,
    },
    state: {
        type: String,
        default: "processing", // Can be 'processing' or 'finished'
    },
    // Add a prop to receive the final session stats
    stats: {
        type: Object,
        required: true,
    },
});

// Defines the custom events that this component can emit.
const emit = defineEmits(["start-new", "view-details", "close"]);

// Computed property to format the detection rate as a percentage.
const formattedDetectionRate = computed(() => {
    if (!props.stats.totalFrames) return "0.00%";
    const rate =
        (props.stats.defectiveProducts / props.stats.totalFrames) * 100;
    return `${rate.toFixed(2)}%`;
});
</script>

<template>
    <TransitionRoot appear :show="isVisible" as="template">
        <Dialog as="div" @close="() => {}" class="relative z-50">
            <!-- Modal Overlay -->
            <TransitionChild
                as="template"
                enter="ease-out duration-300"
                enter-from="opacity-0"
                enter-to="opacity-100"
                leave="ease-in duration-200"
                leave-from="opacity-100"
                leave-to="opacity-0"
            >
                <div
                    class="fixed inset-0 bg-gray-900/50 backdrop-blur dark:bg-black/30"
                />
            </TransitionChild>

            <div class="fixed inset-0 overflow-y-auto">
                <div
                    class="flex min-h-full items-center justify-center p-4 text-center"
                >
                    <!-- Modal Panel -->
                    <TransitionChild
                        as="template"
                        enter="ease-out duration-300"
                        enter-from="opacity-0 scale-95"
                        enter-to="opacity-100 scale-100"
                        leave="ease-in duration-200"
                        leave-from="opacity-100 scale-100"
                        leave-to="opacity-0 scale-95"
                    >
                        <DialogPanel
                            class="w-full max-w-lg transform overflow-hidden rounded-2xl bg-white dark:bg-dark-700 p-8 text-left align-middle shadow-xl transition-all"
                        >
                            <!-- Processing State -->
                            <div
                                v-if="state === 'processing'"
                                class="text-center"
                            >
                                <div
                                    class="flex justify-center items-center mb-4"
                                >
                                    <div
                                        class="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500"
                                    ></div>
                                </div>
                                <h3
                                    class="text-xl font-semibold text-gray-800 dark:text-gray-200"
                                >
                                    Finalizing Session
                                </h3>
                                <p
                                    class="text-gray-500 dark:text-gray-400 mt-2"
                                >
                                    Please wait while we generate the session
                                    report...
                                </p>
                            </div>

                            <!-- Finished State -->
                            <div
                                v-if="state === 'finished'"
                                class="text-center"
                            >
                                <DialogTitle
                                    as="h3"
                                    class="text-lg font-medium leading-6 text-gray-900"
                                >
                                    <!-- Animated Checkmark SVG -->
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        viewBox="0 0 154 154"
                                        fill="none"
                                        class="stroke-current mx-auto size-24 shrink-0 text-green-500"
                                    >
                                        <path
                                            d="M77 141C112.346 141 141 112.346 141 77C141 41.6538 112.346 13 77 13C41.6538 13 13 41.6538 13 77C13 112.346 41.6538 141 77 141Z"
                                            stroke-width="10"
                                            class="checkmark-circle"
                                        ></path>
                                        <path
                                            d="M46 80.2444L63.9556 98.1111L107.067 55"
                                            stroke-width="10"
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            class="checkmark-tick"
                                        ></path>
                                    </svg>
                                </DialogTitle>
                                <h3
                                    class="text-2xl font-bold text-gray-800 dark:text-gray-100 mt-4"
                                >
                                    Session Complete
                                </h3>
                                <!-- Session Stats Summary -->
                                <div
                                    class="mt-4 space-y-2 text-gray-600 dark:text-gray-300"
                                >
                                    <p>
                                        A total of
                                        <strong>{{ stats.totalFrames }}</strong>
                                        frames were analyzed.
                                    </p>
                                    <div class="flex justify-center gap-6">
                                        <span
                                            ><strong class="text-green-500">{{
                                                stats.goodProducts
                                            }}</strong>
                                            Good</span
                                        >
                                        <span
                                            ><strong class="text-red-500">{{
                                                stats.defectiveProducts
                                            }}</strong>
                                            Defects</span
                                        >
                                        <span
                                            ><strong class="text-blue-500">{{
                                                formattedDetectionRate
                                            }}</strong>
                                            Defect Rate</span
                                        >
                                    </div>
                                </div>

                                <!-- Action Buttons -->
                                <div class="mt-6 flex flex-col gap-3">
                                    <button
                                        @click="$emit('start-new')"
                                        class="btn-base btn w-full gap-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                                    >
                                        <font-awesome-icon
                                            icon="fa-solid fa-recycle"
                                        />
                                        Run Another Session
                                    </button>
                                    <Link
                                        :href="route('sessions_scan.index', 1)"
                                    >
                                        <button
                                            class="btn-base btn w-full gap-2 bg-gray-150 text-gray-900 hover:bg-gray-200 focus:bg-gray-200 active:bg-gray-200/80 dark:bg-surface-2 dark:text-dark-50 dark:hover:bg-surface-1 dark:focus:bg-surface-1 dark:active:bg-surface-1/90"
                                        >
                                            <font-awesome-icon
                                                icon="fa-solid fa-folder-open"
                                            />
                                            View Session Details
                                        </button>
                                    </Link>
                                </div>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </div>
        </Dialog>
    </TransitionRoot>
</template>

<style scoped>
/* Keyframe animations for the SVG checkmark */
.checkmark-circle {
    stroke-dasharray: 480px, 480px;
    stroke-dashoffset: 960px;
    animation: draw-circle 1s ease-out forwards;
}

.checkmark-tick {
    stroke-dasharray: 100px, 100px;
    stroke-dashoffset: 200px;
    animation: draw-check 0.8s ease-out 0.5s forwards;
}

@keyframes draw-circle {
    to {
        stroke-dashoffset: 0;
    }
}

@keyframes draw-check {
    to {
        stroke-dashoffset: 0;
    }
}
</style>
