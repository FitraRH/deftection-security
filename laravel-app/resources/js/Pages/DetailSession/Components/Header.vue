<script setup>
import { Link } from "@inertiajs/vue3";
import { route } from "ziggy-js";

// --- Props ---
// This component receives the session data it needs to display.
const props = defineProps({
    session: {
        type: Object,
        required: true,
    },
});

// --- Emits ---
// Defines the events this component can send to the parent page.
const emit = defineEmits(["download", "delete"]);

const handleDownload = () => {
    emit("download", props.session.id);
};

const handleDelete = () => {
    emit("delete", props.session.id);
};
</script>

<template>
    <div
        class="flex flex-col items-start justify-between gap-4 pb-4 border-b border-gray-700 sm:flex-row sm:items-center dark:border-dark-200"
    >
        <div>
            <h1 class="text-2xl font-bold text-gray-900 dark:text-dark-50">
                Session Analysis Details
            </h1>
            <p class="mt-1 text-sm text-gray-500 dark:text-dark-400">
                Session ID: {{ session.id }} • {{ session.date }},
                {{ session.startTime }} - {{ session.endTime }}
            </p>
        </div>
        <div class="flex items-center self-end gap-3 sm:self-auto">
            <button
                @click="handleDownload"
                class="btn-base btn gap-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
            >
                <font-awesome-icon icon="fa-solid fa-download" />
                <span>Download</span>
            </button>
            <button
                @click="handleDelete"
                class="btn-base btn gap-2 text-white transition-colors bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
                <font-awesome-icon icon="fa-solid fa-trash-can" />
                <span>Delete</span>
            </button>
        </div>
    </div>
</template>
