<script setup>
import { ref, computed } from "vue";
import { Link } from "@inertiajs/vue3";
import { route } from "ziggy-js";

// Import Shared Components
import Tablewrapper from "../../Shared/Table/Tablewrapper.vue";
import TableContainer from "../../Shared/Table/TableContainer.vue";
import Table from "../../Shared/Table/Table.vue";
import TableCell from "../../Shared/Table/TableCell.vue";
import TableHeaderCell from "../../Shared/Table/TableHeaderCell.vue";
import RowNotFound from "../../Shared/Table/RowNotFound.vue";
import EllipsisDropdown from "../../Shared/Dropdown/EllipsisDropdown.vue";
import DetailViewList from "../../Shared/Dropdown/ViewItem.vue";
import DeleteItem from "../../Shared/Dropdown/DeleteItem.vue";

// Import Page-Specific Components
import Header from "./Components/Header.vue";
import Summary from "./Components/Summary.vue";
import LineChart from "./Components/LineChart.vue";
import PieChart from "./Components/PieChart.vue";

// Import Page-Specific Components (we can create these next)
// For now, we'll build their content directly into this page for simplicity.

// --- Props ---
// This page will receive the session data from the controller.
const props = defineProps({
    // session: { type: Object, required: true },
});

// --- Mock Data ---
// This represents the data for a single, complete session.
const mockSession = ref({
    id: "SESS-002",
    date: "21/07/2025",
    startTime: "10:01:15",
    endTime: "11:13:27",
    duration: "1 hour 12 minutes",
    total_scans: 340,
    good_scans: 314,
    defected_scans: 26,
    defect_rate: 7.8,
    status: "Completed",
    // The list of individual scans that belong to this session
    scans: [
        {
            id: 101,
            filename: "scan_001.png",
            created_at: "2025-07-21T10:01:15Z",
            status: "good",
            anomaly_score: 0.123,
            original_path:
                "https://placehold.co/100x60/34D399/FFFFFF?text=Good",
        },
        {
            id: 102,
            filename: "scan_002.png",
            created_at: "2025-07-21T10:02:30Z",
            status: "good",
            anomaly_score: 0.234,
            original_path:
                "https://placehold.co/100x60/34D399/FFFFFF?text=Good",
        },
        {
            id: 103,
            filename: "scan_003.png",
            created_at: "2025-07-21T10:03:45Z",
            status: "defect",
            anomaly_score: 0.891,
            original_path:
                "https://placehold.co/100x60/F87171/FFFFFF?text=Defect",
        },
        {
            id: 104,
            filename: "scan_004.png",
            created_at: "2025-07-21T10:05:00Z",
            status: "good",
            anomaly_score: 0.15,
            original_path:
                "https://placehold.co/100x60/34D399/FFFFFF?text=Good",
        },
        // ... more scan objects
    ],
});

const session = computed(() => mockSession.value); // Use mock data for now

// --- Helper Functions ---
const getStatusClass = (status) => {
    if (status === "good")
        return "bg-green-100 text-green-800 dark:bg-green-500/10 dark:text-green-400";
    return "bg-red-100 text-red-800 dark:bg-red-500/10 dark:text-red-400";
};

const formatDecimal = (value) => Number(value).toFixed(3);
</script>

<template>
    <div class="flex flex-col gap-6">
        <!-- Section 1: Session Header -->
        <Header
            :session="session"
            @download="handleDownload"
            @delete="handleDelete"
        />

        <!-- Section 2: Key Metrics Summary -->
        <Summary :session="session" />

        <!-- Section 3: Charts -->
        <div class="gap-6 grid grid-cols-1 sm:grid-cols-2">
            <LineChart :session="session" />
            <PieChart :session="session" />
        </div>

        <!-- Section 3: Scans Table -->
        <Tablewrapper>
            <TableContainer>
                <Table>
                    <template #head>
                        <tr class="group/tr table-tr">
                            <TableHeaderCell label="Image" />
                            <TableHeaderCell label="Timestamp" />
                            <TableHeaderCell label="Status" />
                            <TableHeaderCell label="Score" />
                            <TableHeaderCell
                                label="Actions"
                                :is-sortable="false"
                            />
                        </tr>
                    </template>
                    <template #body>
                        <tr
                            v-for="scan in session.scans"
                            :key="scan.id"
                            class="group/tr table-tr border-b-gray-200 dark:border-b-dark-500"
                        >
                            <TableCell>
                                <div class="flex items-center">
                                    <img
                                        class="flex-shrink-0 rounded-md w-16 h-10 object-cover"
                                        :src="scan.original_path"
                                        alt="Scan Image"
                                    />
                                    <p
                                        class="ml-4 font-medium text-gray-800 dark:text-dark-100"
                                    >
                                        {{ scan.filename }}
                                    </p>
                                </div>
                            </TableCell>
                            <TableCell>
                                <p class="text-gray-800 dark:text-dark-100">
                                    {{
                                        new Date(
                                            scan.created_at
                                        ).toLocaleTimeString()
                                    }}
                                </p>
                            </TableCell>
                            <TableCell>
                                <span
                                    :class="getStatusClass(scan.status)"
                                    class="inline-flex px-2 rounded-full font-semibold text-xs leading-5"
                                >
                                    {{ scan.status.toUpperCase() }}
                                </span>
                            </TableCell>
                            <TableCell>
                                <p
                                    class="font-mono text-gray-800 dark:text-dark-100"
                                >
                                    {{ formatDecimal(scan.anomaly_score) }}
                                </p>
                            </TableCell>
                            <TableCell>
                                <!-- Link to the specific scan detail page -->
                                <Link
                                    :href="
                                        route('sessions_scan.show', {
                                            realtimeSession: session.id,
                                            scan: scan.id,
                                        })
                                    "
                                    class="text-primary-500 hover:text-primary-600"
                                >
                                    View
                                </Link>
                            </TableCell>
                        </tr>
                        <RowNotFound
                            v-if="!session.scans || session.scans.length === 0"
                            label="No scans found for this session"
                        />
                    </template>
                </Table>
            </TableContainer>
        </Tablewrapper>
    </div>
</template>
