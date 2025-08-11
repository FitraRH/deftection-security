<script setup>
import { Link } from "@inertiajs/vue3";
import { route } from "ziggy-js";
import { formatDistanceToNow, format, formatDate } from "date-fns";

// --- Props ---
// This component receives the 5 most recent analyses from the parent dashboard page.
const props = defineProps({
    analyses: {
        type: Array,
        // Mock data is included for standalone development and visualization.
        default: () => [
            {
                id: 1,
                filename: "work.png",
                created_at: "2025-07-20T09:43:49.000000Z",
                status: "good",
                anomaly_score: 0.673,
                original_path:
                    "https://placehold.co/100x60/34D399/FFFFFF?text=Good",
            },
            {
                id: 2,
                filename: "work.png",
                created_at: "2025-07-20T09:40:47.000000Z",
                status: "defect",
                anomaly_score: 0.764,
                original_path:
                    "https://placehold.co/100x60/F87171/FFFFFF?text=Defect",
            },
            {
                id: 3,
                filename: "aokwaoka.png",
                created_at: "2025-07-19T17:58:00.000000Z",
                status: "defect",
                anomaly_score: 0.814,
                original_path:
                    "https://placehold.co/100x60/F87171/FFFFFF?text=Defect",
            },
            {
                id: 4,
                filename: "test.jpg",
                created_at: "2025-07-18T06:02:15.000000Z",
                status: "defect",
                anomaly_score: 0.822,
                original_path:
                    "https://placehold.co/100x60/F87171/FFFFFF?text=Defect",
            },
            {
                id: 5,
                filename: "test.jpg",
                created_at: "2025-07-17T14:27:30.000000Z",
                status: "defect",
                anomaly_score: 0.759,
                original_path:
                    "https://placehold.co/100x60/F87171/FFFFFF?text=Defect",
            },
        ],
    },
});

// --- Helper Functions ---

// Formats a date string into a human-readable format (e.g., "20-07-2025")
const formatDateSimple = (dateString) => {
    if (!dateString) return "N/A";
    try {
        const date = new Date(dateString);
        return format(date, "dd-MM-yyyy");
    } catch (error) {
        return "Invalid date";
    }
};

// Formats a date string into a relative time (e.g., "about 5 hours ago")
const formatRelativeTime = (dateString) => {
    if (!dateString) return "N/A";
    try {
        const date = new Date(dateString);
        return formatDistanceToNow(date, { addSuffix: true });
    } catch (error) {
        return "Invalid date";
    }
};

// Formats a number to a specific decimal precision
const formatDecimal = (value, defaultValue = 0, precision = 3) => {
    const num = Number(value);
    return isNaN(num) ? defaultValue : num.toFixed(precision);
};

// Gets the correct CSS class for the status badge
const getStatusClass = (status) => {
    if (status === "good") {
        return "bg-green-100 text-green-800 dark:bg-green-500/10 dark:text-green-400";
    }
    if (status === "defect") {
        return "bg-red-100 text-red-800 dark:bg-red-500/10 dark:text-red-400";
    }
    return "bg-gray-100 text-gray-800 dark:bg-gray-500/10 dark:text-gray-400";
};
</script>

<template>
    <div class="transition-content flex grow flex-col overflow-x-auto">
        <table class="table w-full text-left rtl:text-right is-hoverable">
            <thead class="group/thead table-thead">
                <tr class="group/tr table-tr">
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Image
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Analysis Date
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Status
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Score
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Actions
                    </th>
                </tr>
            </thead>
            <tbody class="group/tbody table-tbody">
                <tr v-if="analyses.length === 0">
                    <td colspan="5" class="p-4 text-center text-gray-500">
                        No recent analyses found.
                    </td>
                </tr>
                <tr
                    v-for="item in analyses"
                    :key="item.id"
                    class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                >
                    <!-- Image Column -->
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-3 pl-3"
                    >
                        <div class="flex flex-col items-start gap-1">
                            <img
                                :src="item.original_path"
                                alt="Analysis thumbnail"
                                class="h-10 w-16 object-cover rounded-md"
                            />
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{ item.filename }}
                            </p>
                        </div>
                    </td>
                    <!-- Analysis Date Column -->
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-3 pl-3"
                    >
                        <p class="font-medium text-gray-800 dark:text-dark-100">
                            {{ formatDateSimple(item.created_at) }}
                        </p>

                        <p class="text-xs text-gray-800 dark:text-dark-100">
                            {{ formatRelativeTime(item.created_at) }}
                        </p>
                    </td>
                    <!-- Status Column -->
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-3 pl-3"
                    >
                        <span
                            class="inline-block px-2 py-1 rounded font-semibold text-xs uppercase"
                            :class="getStatusClass(item.status)"
                        >
                            {{ item.status }}
                        </span>
                    </td>
                    <!-- Score Column -->
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-3 pl-3"
                    >
                        <p
                            class="font-mono text-sm text-gray-800 dark:text-dark-100"
                        >
                            {{ formatDecimal(item.anomaly_score, "N/A", 3) }}
                        </p>
                    </td>
                    <!-- Actions Column -->
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-3 pl-3"
                    >
                        <Link :href="route('scans.show', item.id)">
                            <div class="flex justify-center pr-3">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 24 24"
                                    fill="currentColor"
                                    class="size-5 text-primary-600 hover:text-primary-600/70 focus:text-primary-600/70 dark:hover:text-primary-400/70 dark:focus:text-primary-400/70 dark:text-primary-400 transition-colors duration-300"
                                >
                                    <path
                                        d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"
                                    />
                                    <path
                                        fill-rule="evenodd"
                                        d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z"
                                        clip-rule="evenodd"
                                    />
                                </svg>
                            </div>
                        </Link>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</template>
