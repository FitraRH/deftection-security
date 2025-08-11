<script setup>
import {
    Dialog,
    DialogPanel,
    DialogTitle,
    TransitionRoot,
    TransitionChild,
} from "@headlessui/vue";

// --- Props ---
defineProps({
    // Controls the visibility of the modal
    show: {
        type: Boolean,
        default: false,
    },
    // The main title of the modal (e.g., "Reset Settings?")
    title: {
        type: String,
        required: true,
    },
    // The descriptive text explaining the action
    message: {
        type: String,
        required: true,
    },
    // The text for the confirmation button (e.g., "Confirm Reset")
    confirmText: {
        type: String,
        default: "Confirm",
    },
    // The icon for the modal header
    icon: {
        type: String,
        default: "fa-solid fa-triangle-exclamation",
    },
    // The color theme for the icon and confirm button (e.g., 'error' for red)
    variant: {
        type: String,
        default: "warning", // can be 'warning', 'error', 'info'
    },
});

// --- Emits ---
const emit = defineEmits(["close", "confirm"]);
</script>

<template>
    <TransitionRoot appear :show="show" as="template">
        <Dialog as="div" @close="emit('close')" class="relative z-50">
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
                            class="w-full max-w-md transform overflow-hidden rounded-2xl bg-white dark:bg-dark-700 p-6 text-left align-middle shadow-xl transition-all"
                        >
                            <div class="flex items-start gap-4">
                                <!-- Icon -->
                                <div
                                    class="flex-shrink-0 h-12 w-12 flex items-center justify-center rounded-full"
                                    :class="{
                                        'bg-red-100 dark:bg-red-900/50':
                                            variant === 'error',
                                        'bg-yellow-100 dark:bg-yellow-900/50':
                                            variant === 'warning',
                                    }"
                                >
                                    <font-awesome-icon
                                        :icon="icon"
                                        class="h-6 w-6"
                                        :class="{
                                            'text-red-600 dark:text-red-400':
                                                variant === 'error',
                                            'text-yellow-600 dark:text-yellow-400':
                                                variant === 'warning',
                                        }"
                                    />
                                </div>

                                <!-- Content -->
                                <div class="flex-1">
                                    <DialogTitle
                                        as="h3"
                                        class="text-lg font-medium leading-6 text-gray-900 dark:text-gray-100"
                                    >
                                        {{ title }}
                                    </DialogTitle>
                                    <div class="mt-2">
                                        <p
                                            class="text-sm text-gray-500 dark:text-gray-400"
                                        >
                                            {{ message }}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <!-- Action Buttons -->
                            <div class="mt-6 flex justify-end gap-3">
                                <button
                                    type="button"
                                    class="btn-base btn bg-gray-200 text-gray-800 hover:bg-gray-300 dark:bg-dark-600 dark:text-gray-200 dark:hover:bg-dark-500"
                                    @click="emit('close')"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="button"
                                    class="btn-base btn text-white"
                                    :class="{
                                        'this:error bg-this hover:bg-this-darker':
                                            variant === 'error',
                                        'this:warning bg-this hover:bg-this-darker':
                                            variant === 'warning',
                                    }"
                                    @click="emit('confirm')"
                                >
                                    {{ confirmText }}
                                </button>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </div>
        </Dialog>
    </TransitionRoot>
</template>
