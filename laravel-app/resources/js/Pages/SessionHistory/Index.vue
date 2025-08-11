<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { router } from "@inertiajs/vue3";
import { useDebounceFn } from "@vueuse/core";
import { usePolling } from "@/composables/usePolling";
import { route } from "ziggy-js";
import { useForm } from "@inertiajs/vue3";
import { successToast, errorToast, deleteConfirmDialog } from "@/utils/swal";
import { Popover } from "@headlessui/vue";

// Import shared components
import Table from "../../Shared/Table/Table.vue";
import Tablewrapper from "../../Shared/Table/Tablewrapper.vue";
import TableToolbar from "../../Shared/Table/TableToolbar.vue";
import TableContainer from "../../Shared/Table/TableContainer.vue";
import ToolbarSearch from "../../Shared/TableToolbar/ToolbarSearch.vue";
import FilterDateRange from "../../Shared/TableToolbar/FilterDateRange.vue";
import TableHeaderCell from "../../Shared/Table/TableHeaderCell.vue";
import ForceRefreshButton from "../../Shared/Table/ForceRefreshButton.vue";
import LiveMonitorToggle from "../../Shared/Table/LiveMonitorToggle.vue";
import TableCell from "../../Shared/Table/TableCell.vue";
import RowNotFound from "../../Shared/Table/RowNotFound.vue";
import TableFooter from "../../Shared/Table/TableFooter.vue";
import EllipsisDropdown from "../../Shared/Dropdown/EllipsisDropdown.vue";
import DetailViewList from "../../Shared/Dropdown/ViewItem.vue";
import DeleteItem from "../../Shared/Dropdown/DeleteItem.vue";
import FilterPopoverButton from "../../Shared/Popover/FilterPopoverButton.vue";
import FilterPopoverPanel from "../../Shared/Popover/FilterPopoverPanel.vue";
import FilterPopoverCheckbox from "../../Shared/Popover/FilterPopoverCheckbox.vue";
import FilterResetButton from "../../Shared/TableToolbar/FilterResetButton.vue";

// --- Props ---
const props = defineProps({
    sessions: { type: Array, required: true },
    filters: { type: Object, required: true },
    meta: { type: Object, default: () => ({}) },
    initialChecksum: { type: String, default: "" },
    userCan: {
        type: Object,
        default: () => ({
            viewAllSessions: false,
            filterByUser: false,
        }),
    },
});

// --- State ---
const currentFilters = computed(() => props.filters?.current || {});
const filterOptions = computed(() => props.filters?.options || {});
const searchTerm = ref(currentFilters.value.search || "");
const dateFrom = ref(currentFilters.value.dateFrom || "");
const dateTo = ref(currentFilters.value.dateTo || "");

// For polling updates only
const pollingSessions = ref([]);
const pollingFilters = ref({});
const pollingMeta = ref({});

// Delete state
const isDeletingSession = ref(false);

// Loading states for UI control
const isTableLoading = ref(false);

// Determine which data to show: polling data if available, otherwise props
const displaySessions = computed(() => {
    return pollingSessions.value.length > 0
        ? pollingSessions.value
        : props.sessions;
});

const displayMeta = computed(() => {
    return Object.keys(pollingMeta.value).length > 0
        ? pollingMeta.value
        : props.meta;
});

const displayFilterOptions = computed(() => {
    return Object.keys(pollingFilters.value).length > 0
        ? pollingFilters.value
        : filterOptions.value;
});

// Check if any operations are in progress
const isAnyOperationInProgress = computed(() => {
    return (
        isTableLoading.value ||
        isPollingLoading.value ||
        isDeletingSession.value
    );
});

// --- Polling Logic ---
const {
    isPolling,
    isLoading: isPollingLoading,
    error: pollingError,
    isManuallyPaused,
    canPoll,
    isOnline,
    visibility,
    togglePolling,
    forceRefresh,
    initializeChecksum,
    getStatus,
} = usePolling({
    // Custom endpoints for this component
    checkUrl: route("sessions.index.check"),
    dataUrl: route("sessions.index.api"),
    forceRefreshUrl: route("sessions.index.refresh"),

    // Polling configuration
    interval: 20000,
    enabled: true,
    pauseOnTabHidden: true,
    pauseOnOffline: true,
    debug: false,
    retryAttempts: 3,
    retryDelay: 10000,

    // Custom pause condition
    customPauseCondition: () => {
        return true;
    },

    // Callbacks
    onDataUpdate: (data) => {
        // Only update polling state, don't interfere with navigation
        pollingSessions.value = data.sessions || [];
        pollingFilters.value = data.filters?.options || {};
        pollingMeta.value = data.meta || {};

        // Reinitialize filter options with new data
        initializeFilters();

        console.log("Session polling detected changes and updated data!");
    },
    onError: (error) => {
        console.error("Session polling error:", error);
    },
});

// --- Filter Options ---
const sessionStatusOptions = ref([]);
const userOptions = ref([]);
const roleOptions = ref([]);

// Initialize filter options
const initializeFilters = () => {
    const options = displayFilterOptions.value;

    const mapOptions = (options, selectedKey) =>
        options?.map((option) => ({
            ...option,
            selected: (currentFilters.value[selectedKey] || []).includes(
                option.value
            ),
        })) || [];

    sessionStatusOptions.value = mapOptions(
        options.sessionStatuses,
        "sessionStatuses"
    );

    if (props.userCan.filterByUser) {
        userOptions.value = mapOptions(options.users, "users");
        roleOptions.value = mapOptions(options.roles, "roles");
    }
};

// Add date filter handler
const handleDateFilterChange = () => {
    if (!isAnyOperationInProgress.value) {
        setTimeout(applyFilters, 10);
    }
};

// Watch for prop changes to reinitialize filters
watch(
    () => props.filters,
    () => {
        initializeFilters();
    },
    { deep: true }
);

// Watch search term to update from current filters
watch(
    () => currentFilters.value.search,
    (newSearch) => {
        searchTerm.value = newSearch || "";
    }
);

// Watch for date filter changes from current filters
watch(
    () => [currentFilters.value.dateFrom, currentFilters.value.dateTo],
    ([newDateFrom, newDateTo]) => {
        dateFrom.value = newDateFrom || "";
        dateTo.value = newDateTo || "";
    }
);

// Initialize component
onMounted(() => {
    initializeFilters();
    initializeChecksum(props.initialChecksum);
});

// --- Computed Properties ---
const selectedSessionStatusCount = computed(
    () => sessionStatusOptions.value.filter((option) => option.selected).length
);
const selectedUsersCount = computed(
    () => userOptions.value.filter((option) => option.selected).length
);
const selectedRolesCount = computed(
    () => roleOptions.value.filter((option) => option.selected).length
);
const hasActiveFilters = computed(
    () =>
        selectedSessionStatusCount.value > 0 ||
        selectedUsersCount.value > 0 ||
        selectedRolesCount.value > 0 ||
        searchTerm.value.length > 0 ||
        dateFrom.value.length > 0 ||
        dateTo.value.length > 0
);
const selectedSessionStatuses = computed(() =>
    sessionStatusOptions.value
        .filter((option) => option.selected)
        .map((option) => option.value)
);
const selectedUsers = computed(() =>
    userOptions.value
        .filter((option) => option.selected)
        .map((option) => parseInt(option.value))
);
const selectedRoles = computed(() =>
    roleOptions.value
        .filter((option) => option.selected)
        .map((option) => option.value)
);

// --- Helper Functions ---
const getStatusClass = (status) => {
    switch (status?.toLowerCase()) {
        case "active":
            return "text-blue-500";
        case "completed":
            return "text-green-500";
        case "paused":
            return "text-yellow-500";
        case "stopped":
            return "text-red-500";
        default:
            return "text-gray-500";
    }
};

const formatDuration = (seconds) => {
    if (!seconds || seconds === 0) return "0s";

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
};

// --- Delete Functionality ---
const handleDeleteSession = async (session) => {
    const result = await deleteConfirmDialog(`session "${session.id}"`);

    if (result.isConfirmed) {
        await performDelete(session);
    }
};

const performDelete = async (session) => {
    isDeletingSession.value = true;

    const form = useForm({});

    form.delete(route("sessions.destroy", session.id), {
        preserveState: true,
        preserveScroll: true,
        onSuccess: () => {
            handleDeleteSuccess(session.id);
            forceRefresh();
        },
        onError: (errors) => {
            handleDeleteError(errors);
        },
        onFinish: () => {
            isDeletingSession.value = false;
        },
    });
};

const handleDeleteSuccess = (sessionId) => {
    console.log("Session deleted successfully:", sessionId);
    successToast("Session deleted successfully!");
};

const handleDeleteError = (errors) => {
    console.error("Delete operation failed:", errors);

    let errorMessage =
        "An unexpected error occurred while deleting the session.";

    if (errors.message) {
        errorMessage = errors.message;
    } else if (errors.error) {
        errorMessage = errors.error;
    }

    errorToast(errorMessage);
};

// --- Build Filter Parameters ---
const buildFilterParams = (customParams = {}) => {
    const params = {
        search: searchTerm.value || undefined,
        session_statuses:
            selectedSessionStatuses.value.length > 0
                ? selectedSessionStatuses.value
                : undefined,
        users: selectedUsers.value.length > 0 ? selectedUsers.value : undefined,
        roles: selectedRoles.value.length > 0 ? selectedRoles.value : undefined,
        date_from: dateFrom.value || undefined,
        date_to: dateTo.value || undefined,
        sort_by: currentFilters.value.sortBy || undefined,
        sort_dir: currentFilters.value.sortDir || undefined,
        per_page: currentFilters.value.perPage || undefined,
        page: 1,
        ...customParams,
    };

    // Remove undefined values
    Object.keys(params).forEach((key) => {
        if (params[key] === undefined) delete params[key];
    });

    return params;
};

// Navigate with filters helper
const navigateWithFilters = (customParams = {}) => {
    // Set loading state
    isTableLoading.value = true;

    // Clear polling data to show fresh server data
    pollingSessions.value = [];
    pollingFilters.value = {};
    pollingMeta.value = {};

    const params = buildFilterParams(customParams);

    router.get(route("sessions.index"), params, {
        preserveState: true,
        preserveScroll: true,
        replace: true,
        only: ["sessions", "filters", "meta", "initialChecksum"],
        onSuccess: (page) => {
            // Update checksum for polling
            if (page.props.initialChecksum) {
                initializeChecksum(page.props.initialChecksum);
            }

            // Reinitialize filters with new data after props are updated
            setTimeout(() => {
                initializeFilters();
            }, 50);
        },
        onError: (errors) => {
            console.error("Navigation failed:", errors);
        },
        onFinish: () => {
            isTableLoading.value = false;
        },
    });
};

// Main filter application
const applyFilters = () => navigateWithFilters();

// Filter methods
const clearFilter = (optionsRef) => {
    optionsRef.value.forEach((option) => (option.selected = false));
    applyFilters();
};

const clearSessionStatusFilters = () => clearFilter(sessionStatusOptions);
const clearUserFilters = () => clearFilter(userOptions);
const clearRoleFilters = () => clearFilter(roleOptions);

const resetAllFilters = () => {
    clearFilter(sessionStatusOptions);
    clearFilter(userOptions);
    clearFilter(roleOptions);
    searchTerm.value = "";
    dateFrom.value = "";
    dateTo.value = "";
    applyFilters();
};

// Search and filter change handlers
const debouncedSearch = useDebounceFn(applyFilters, 300);
const handleSearchChange = () => {
    if (!isAnyOperationInProgress.value) {
        debouncedSearch();
    }
};
const handleFilterChange = () => {
    if (!isAnyOperationInProgress.value) {
        setTimeout(applyFilters, 10);
    }
};

// Pagination and sorting
const goToPage = (page) => {
    if (!isAnyOperationInProgress.value) {
        navigateWithFilters({ page });
    }
};

const changePerPage = (perPage) => {
    if (!isAnyOperationInProgress.value) {
        navigateWithFilters({ per_page: perPage, page: 1 });
    }
};

const sortBy = (column) => {
    if (!isAnyOperationInProgress.value) {
        const sortDir =
            currentFilters.value.sortBy === column
                ? currentFilters.value.sortDir === "asc"
                    ? "desc"
                    : "asc"
                : "asc";
        navigateWithFilters({ sort_by: column, sort_dir: sortDir, page: 1 });
    }
};

// Header configuration
const headerConfig = [
    {
        label: "session id",
        field: "id",
        sortable: true,
    },
    {
        label: "date",
        field: "session_start",
        sortable: true,
    },
    {
        label: "duration",
        field: "duration_seconds",
        sortable: true,
    },
    {
        label: "total scans",
        field: "total_frames_processed",
        sortable: true,
    },
    {
        label: "defect rate",
        field: "defect_rate",
        sortable: true,
    },
    {
        label: "status",
        field: "session_status",
        sortable: true,
    },
    { label: "actions", sortable: false },
];

// Add user column for admins who can view all sessions
if (props.userCan.viewAllSessions) {
    headerConfig.splice(-1, 0, {
        label: "user",
        sortable: false,
    });
}
</script>

<template>
    <Tablewrapper>
        <div class="flex justify-between items-center mb-4">
            <!-- Polling Toggle Button -->
            <LiveMonitorToggle
                :isPolling="isPolling"
                :isPollingLoading="isPollingLoading"
                :isManuallyPaused="isManuallyPaused"
                :isOnline="isOnline"
                :visibility="visibility"
                @click="togglePolling"
            />
            <!-- Force Refresh Button -->
            <ForceRefreshButton
                :isPollingLoading="isPollingLoading"
                :isOnline="isOnline"
                :disabled="isTableLoading"
                @click="forceRefresh"
            />
        </div>

        <TableToolbar
            title="Session History"
            description="History of your real-time detection sessions with detailed results."
        >
            <template #left>
                <ToolbarSearch
                    v-model="searchTerm"
                    placeholder="Search by Session ID..."
                    :disabled="isAnyOperationInProgress"
                    @input="handleSearchChange"
                />

                <!-- Date Range Filter -->
                <FilterDateRange
                    v-model:dateFrom="dateFrom"
                    v-model:dateTo="dateTo"
                    :disabled="isAnyOperationInProgress"
                    @change="handleDateFilterChange"
                />

                <!-- Status Filter -->
                <Popover class="relative">
                    <FilterPopoverButton
                        label="Status"
                        :selected-options="selectedSessionStatusCount"
                        :disabled="isAnyOperationInProgress"
                    >
                        <template #icon>
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24"
                                fill="currentColor"
                                class="size-4"
                            >
                                <path
                                    fill-rule="evenodd"
                                    d="M3.792 2.938A49.069 49.069 0 0 1 12 2.25c2.797 0 5.54.236 8.209.688a1.857 1.857 0 0 1 1.541 1.836v1.044a3 3 0 0 1-.879 2.121l-6.182 6.182a1.5 1.5 0 0 0-.439 1.061v2.927a3 3 0 0 1-1.658 2.684l-1.757.878A.75.75 0 0 1 9.75 21v-5.818a1.5 1.5 0 0 0-.44-1.06L3.13 7.938a3 3 0 0 1-.879-2.121V4.774c0-.897.64-1.683 1.542-1.836Z"
                                    clip-rule="evenodd"
                                />
                            </svg>
                        </template>
                    </FilterPopoverButton>
                    <FilterPopoverPanel
                        :show-clear-button="selectedSessionStatusCount > 0"
                        :selected-options="selectedSessionStatusCount"
                        @click-clear="clearSessionStatusFilters"
                    >
                        <FilterPopoverCheckbox
                            v-for="status in sessionStatusOptions"
                            :key="status.value"
                            v-model:checked="status.selected"
                            :id="`session-status-${status.value}`"
                            :value="status.value"
                            :label="status.label"
                            :count="status.count"
                            :disabled="isAnyOperationInProgress"
                            @change="handleFilterChange"
                        />
                        <div
                            v-if="sessionStatusOptions.length === 0"
                            class="px-3 py-2 text-xs"
                        >
                            No session statuses found
                        </div>
                    </FilterPopoverPanel>
                </Popover>

                <!-- User Filter (Admin Only) -->
                <Popover v-if="userCan.filterByUser" class="relative">
                    <FilterPopoverButton
                        label="User"
                        :selected-options="selectedUsersCount"
                        :disabled="isAnyOperationInProgress"
                    >
                        <template #icon>
                            <font-awesome-icon icon="fa-solid fa-user" />
                        </template>
                    </FilterPopoverButton>
                    <FilterPopoverPanel
                        :show-clear-button="selectedUsersCount > 0"
                        :selected-options="selectedUsersCount"
                        @click-clear="clearUserFilters"
                    >
                        <FilterPopoverCheckbox
                            v-for="user in userOptions"
                            :key="user.value"
                            v-model:checked="user.selected"
                            :id="`user-${user.value}`"
                            :value="user.value"
                            :label="user.label"
                            :count="user.count"
                            :disabled="isAnyOperationInProgress"
                            @change="handleFilterChange"
                        />
                        <div
                            v-if="userOptions.length === 0"
                            class="px-3 py-2 text-xs"
                        >
                            No users found
                        </div>
                    </FilterPopoverPanel>
                </Popover>

                <!-- Role Filter (Admin Only) -->
                <Popover v-if="userCan.filterByUser" class="relative">
                    <FilterPopoverButton
                        label="Role"
                        :selected-options="selectedRolesCount"
                        :disabled="isAnyOperationInProgress"
                    >
                        <template #icon>
                            <font-awesome-icon icon="fa-solid fa-key" />
                        </template>
                    </FilterPopoverButton>
                    <FilterPopoverPanel
                        :show-clear-button="selectedRolesCount > 0"
                        :selected-options="selectedRolesCount"
                        @click-clear="clearRoleFilters"
                    >
                        <FilterPopoverCheckbox
                            v-for="role in roleOptions"
                            :key="role.value"
                            v-model:checked="role.selected"
                            :id="`role-${role.value}`"
                            :value="role.value"
                            :label="role.label"
                            :count="role.count"
                            :disabled="isAnyOperationInProgress"
                            @change="handleFilterChange"
                        />
                        <div
                            v-if="roleOptions.length === 0"
                            class="px-3 py-2 text-xs"
                        >
                            No roles found
                        </div>
                    </FilterPopoverPanel>
                </Popover>

                <FilterResetButton
                    v-if="hasActiveFilters"
                    label="Reset Filters"
                    :disabled="isAnyOperationInProgress"
                    @click="resetAllFilters"
                />
            </template>
        </TableToolbar>

        <TableContainer>
            <!-- Loading overlay -->
            <div
                v-if="isTableLoading"
                class="z-10 absolute inset-0 flex justify-center items-center bg-white/10 dark:bg-gray-900/10"
            >
                <div
                    class="flex items-center space-x-2 text-gray-600 dark:text-gray-300"
                >
                    <svg
                        class="w-5 h-5 animate-spin"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                    >
                        <circle
                            class="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            stroke-width="4"
                        ></circle>
                        <path
                            class="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                    </svg>
                    <span>Loading...</span>
                </div>
            </div>

            <Table>
                <template #head>
                    <tr class="group/tr table-tr">
                        <TableHeaderCell
                            v-for="header in headerConfig"
                            :key="header.label"
                            :label="header.label"
                            :is-sortable="
                                header.sortable && !isAnyOperationInProgress
                            "
                            :is-active="currentFilters.sortBy === header.field"
                            :sort-dir="currentFilters.sortDir"
                            :disabled="isAnyOperationInProgress"
                            @click="header.sortable && sortBy(header.field)"
                        />
                    </tr>
                </template>
                <template #body>
                    <tr
                        v-for="session in displaySessions"
                        v-if="displaySessions.length > 0"
                        :key="session.id || Math.random()"
                        class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                        :class="{ 'opacity-50': isTableLoading }"
                    >
                        <!-- Session ID Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{ session.id }}
                            </p>
                        </TableCell>

                        <!-- Date Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ session.session_date }}
                            </p>
                        </TableCell>

                        <!-- Duration Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ formatDuration(session.duration_seconds) }}
                            </p>
                        </TableCell>

                        <!-- Total Scans Column -->
                        <TableCell>
                            <p
                                class="font-mono text-gray-800 dark:text-dark-100"
                            >
                                {{ session.total_scans || 0 }}
                            </p>
                        </TableCell>

                        <!-- Defect Rate Column -->
                        <TableCell>
                            <p
                                class="font-mono font-semibold"
                                :class="
                                    parseFloat(session.defect_rate) > 10
                                        ? 'text-red-500'
                                        : 'text-gray-800 dark:text-dark-100'
                                "
                            >
                                {{ session.defect_rate }}
                            </p>
                        </TableCell>

                        <!-- Status Column -->
                        <TableCell>
                            <span
                                :class="[
                                    'inline-flex px-2 rounded-full font-semibold text-xs uppercase leading-5',
                                    session.status?.toLowerCase() === 'active'
                                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-500/10 dark:text-blue-400'
                                        : session.status?.toLowerCase() ===
                                          'completed'
                                        ? 'bg-green-100 text-green-800 dark:bg-green-500/10 dark:text-green-400'
                                        : session.status?.toLowerCase() ===
                                          'paused'
                                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-500/10 dark:text-yellow-400'
                                        : session.status?.toLowerCase() ===
                                          'stopped'
                                        ? 'bg-red-100 text-red-800 dark:bg-red-500/10 dark:text-red-400'
                                        : 'bg-gray-100 text-gray-800 dark:bg-gray-500/10 dark:text-gray-400',
                                ]"
                            >
                                {{ session.status || "unknown" }}
                            </span>
                        </TableCell>

                        <!-- User Column (only for admins) -->
                        <TableCell v-if="userCan.viewAllSessions">
                            <div>
                                <p
                                    class="font-medium text-gray-800 dark:text-dark-100 capitalize"
                                >
                                    {{ session?.username || "Unknown" }}
                                </p>
                                <p
                                    class="mt-1 text-gray-400 dark:text-dark-300 text-xs uppercase"
                                >
                                    {{ session?.user_role || "N/A" }}
                                </p>
                            </div>
                        </TableCell>

                        <!-- Actions Column -->
                        <TableCell>
                            <EllipsisDropdown
                                :disabled="isAnyOperationInProgress"
                            >
                                <DetailViewList
                                    label="View Scans"
                                    :href="
                                        route('sessions_scan.index', {
                                            realtimeSession: session.id,
                                        })
                                    "
                                    title="View Session Scans"
                                    :disabled="isAnyOperationInProgress"
                                />
                                <DeleteItem
                                    label="Delete Session"
                                    title="Delete Session"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleDeleteSession(session)"
                                />
                            </EllipsisDropdown>
                        </TableCell>
                    </tr>

                    <RowNotFound
                        v-if="displaySessions.length === 0 && !isTableLoading"
                        label="No sessions found"
                    />
                </template>
            </Table>

            <TableFooter
                :meta="displayMeta"
                :currentFilters="currentFilters"
                :perPageOptions="
                    displayFilterOptions.perPageOptions || [5, 10, 25, 50]
                "
                :disabled="isAnyOperationInProgress"
                @goToPage="goToPage"
                @changePerPage="changePerPage"
            />
        </TableContainer>
    </Tablewrapper>
</template>
