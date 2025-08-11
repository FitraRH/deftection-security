<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { router } from "@inertiajs/vue3";
import { useDebounceFn } from "@vueuse/core";
import { usePolling } from "@/composables/usePolling";
import { route } from "ziggy-js";
import { useForm } from "@inertiajs/vue3";
import { successToast, errorToast, deleteConfirmDialog } from "@/utils/swal";

// Import child components
import Table from "../../../Shared/Table/Table.vue";
import Tablewrapper from "../../../Shared/Table/Tablewrapper.vue";
import TableToolbar from "../../../Shared/Table/TableToolbar.vue";
import TableContainer from "../../../Shared/Table/TableContainer.vue";
import ToolbarSearch from "../../../Shared/TableToolbar/ToolbarSearch.vue";
import TableHeaderCell from "../../../Shared/Table/TableHeaderCell.vue";
import ForceRefreshButton from "../../../Shared/Table/ForceRefreshButton.vue";
import LiveMonitorToggle from "../../../Shared/Table/LiveMonitorToggle.vue";
import FilterResetButton from "../../../Shared/TableToolbar/FilterResetButton.vue";
import TableCell from "../../../Shared/Table/TableCell.vue";
import RowNotFound from "../../../Shared/Table/RowNotFound.vue";
import TableFooter from "../../../Shared/Table/TableFooter.vue";
import EllipsisDropdown from "../../../Shared/Dropdown/EllipsisDropdown.vue";
import CreateButton from "../../../Shared/TableToolbar/CreateButton.vue";
import DefectTypeCreateModal from "./Components/DefectTypeCreateModal.vue";
import EditItem from "../../../Shared/Dropdown/EditItem.vue";
import DefectTypeEditModal from "./Components/DefectTypeEditModal.vue";
import DeleteItem from "../../../Shared/Dropdown/DeleteItem.vue";

const props = defineProps({
    defect_types: { type: Array, required: true },
    filters: { type: Object, required: true },
    meta: { type: Object, default: () => ({}) },
    initialChecksum: { type: String, default: "" },
});

console.log("DefectType Index component initialized with props:", props);

// Theme store for SweetAlert styling - removed as swal utils handle this

// Use props directly for navigation, reactive refs for polling updates
const currentFilters = computed(() => props.filters.current || {});
const filterOptions = computed(() => props.filters.options || {});
const searchTerm = ref(currentFilters.value.search || "");

// For polling updates only
const pollingDefectTypes = ref([]);
const pollingFilters = ref({});
const pollingMeta = ref({});

// Modal state management
const isAnyModalOpen = ref(false);
const openModals = ref(new Set()); // Track multiple modals if needed

// Specific modal states
const isCreateModalOpen = ref(false);
const isEditModalOpen = ref(false);
const editDefectTypeSlug = ref(null);

// Delete state
const isDeletingDefectType = ref(false);

// Loading states for UI control
const isTableLoading = ref(false);

// Determine which data to show: polling data if available, otherwise props
const displayDefectTypes = computed(() => {
    return pollingDefectTypes.value.length > 0
        ? pollingDefectTypes.value
        : props.defect_types;
});

console.log(
    "Display defect types computed property initialized with data:",
    displayDefectTypes.value
);

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
        isDeletingDefectType.value
    );
});

// CLEAN REUSABLE POLLING with modal awareness
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
    checkUrl: route("defect_types.index.check"),
    dataUrl: route("defect_types.index.api"),
    forceRefreshUrl: route("defect_types.index.refresh"),

    // Polling configuration
    interval: 20000,
    enabled: true,
    pauseOnTabHidden: true,
    pauseOnOffline: true,
    debug: false,
    retryAttempts: 3,
    retryDelay: 10000,

    // Custom pause condition - pause when modal is open
    customPauseCondition: () => {
        // Pause polling when any modal is open
        const shouldPause = isAnyModalOpen.value;
        if (shouldPause) {
            console.log("Polling paused: Modal is open");
        }
        return !shouldPause; // Return true to allow polling, false to pause
    },

    // Callbacks
    onDataUpdate: (data) => {
        // Only update polling state, don't interfere with navigation
        pollingDefectTypes.value = data.defect_types || [];
        pollingFilters.value = data.filters?.options || {};
        pollingMeta.value = data.meta || {};

        console.log("Polling detected changes and updated data!");
    },
    onError: (error) => {
        console.error("Polling error:", error);
        // Could add toast notifications, etc.
    },
});

// Watch search term to update from current filters
watch(
    () => currentFilters.value.search,
    (newSearch) => {
        searchTerm.value = newSearch || "";
    }
);

// Initialize component
onMounted(() => {
    initializeChecksum(props.initialChecksum);
});

// Computed properties
const hasActiveFilters = computed(() => searchTerm.value.length > 0);

// Modal state management functions
const handleModalOpen = (modalId = "default") => {
    openModals.value.add(modalId);
    isAnyModalOpen.value = true;
};

const handleModalClose = (modalId = "default") => {
    openModals.value.delete(modalId);
    isAnyModalOpen.value = openModals.value.size > 0;
    console.log(
        `Modal ${modalId} closed. Polling will ${
            isAnyModalOpen.value ? "remain paused" : "resume"
        }.`
    );
};

// Create modal specific handlers
const handleCreateModalOpen = () => {
    isCreateModalOpen.value = true;
    handleModalOpen("create");
};

const handleCreateModalClose = () => {
    isCreateModalOpen.value = false;
    handleModalClose("create");
};

// Edit modal specific handlers
const handleEditModalOpen = (slug) => {
    editDefectTypeSlug.value = slug;
    isEditModalOpen.value = true;
    handleModalOpen(`edit-${slug}`);
};

const handleEditModalClose = () => {
    const slug = editDefectTypeSlug.value;
    isEditModalOpen.value = false;
    editDefectTypeSlug.value = null;
    handleModalClose(`edit-${slug}`);
};

// Delete functionality
const handleDeleteDefectType = async (product) => {
    // Show confirmation dialog using clean swal utils
    const result = await deleteConfirmDialog(`"${product.name}"`);

    if (result.isConfirmed) {
        await performDelete(product);
    }
};

// Perform the actual deletion
const performDelete = async (product) => {
    isDeletingDefectType.value = true;

    const form = useForm({});

    form.delete(route("defect_types.destroy", product.slug), {
        preserveState: true,
        preserveScroll: true,
        onSuccess: () => {
            handleDeleteSuccess(product.name);
        },
        onError: (errors) => {
            handleDeleteError(errors);
        },
        onFinish: () => {
            isDeletingDefectType.value = false;
        },
    });
};

// Handle successful deletion
const handleDeleteSuccess = (productName) => {
    console.log("Defect Type deleted successfully:", productName);

    successToast("Defect Type deleted successfully!");
};

// Handle deletion error
const handleDeleteError = (errors) => {
    console.error("Delete operation failed:", errors);

    let errorMessage =
        "An unexpected error occurred while deleting the product.";

    // Check for specific error messages
    if (errors.message) {
        errorMessage = errors.message;
    } else if (errors.error) {
        errorMessage = errors.error;
    }

    errorToast(errorMessage);
};

// Build filter parameters helper
const buildFilterParams = (customParams = {}) => {
    const params = {
        search: searchTerm.value || undefined,
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
    pollingDefectTypes.value = [];
    pollingFilters.value = {};
    pollingMeta.value = {};

    router.get(route("defect_types.index"), buildFilterParams(customParams), {
        preserveState: true,
        preserveScroll: true,
        replace: true,
        only: ["defect_types", "filters", "meta", "initialChecksum"],
        onSuccess: (page) => {
            // Update checksum for polling
            if (page.props.initialChecksum) {
                initializeChecksum(page.props.initialChecksum);
            }

            console.log("Navigation successful, data updated");
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
const resetAllFilters = () => {
    searchTerm.value = "";
    applyFilters();
};

// Search and filter change handlers
const debouncedSearch = useDebounceFn(applyFilters, 300);
const handleSearchChange = () => {
    if (!isAnyOperationInProgress.value) {
        debouncedSearch();
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

// Event handlers for child components
const handleDefectTypeOperation = ({ type, action, data, error, message }) => {
    console.log(`Defect Type ${action} operation:`, { type, message });

    switch (type) {
        case "success":
            switch (action) {
                case "create":
                    console.log("Defect Type created successfully:", data);
                    break;

                case "update":
                    console.log("Defect Type updated successfully:", data);
                    break;

                case "fetch":
                    console.log("Defect Type fetched successfully:", data);
                    break;
            }
            break;

        case "error":
            switch (action) {
                case "create":
                    console.error("Failed to create product:", error);
                    break;

                case "update":
                    console.error("Failed to update product:", error);
                    break;

                case "fetch":
                    console.error("Failed to fetch product:", error);
                    break;
            }
            break;

        case "finish":
            switch (action) {
                case "create":
                    console.log("Create operation finished");
                    break;

                case "update":
                    console.log("Update operation finished");
                    break;
            }
            break;

        default:
            console.warn("Unknown operation type:", type);
    }
};

// --- Scroll Logic ---
const tableRef = ref(null);

const handleDropdownOpen = () => {
    console.log("Dropdown opened, scrolling to bottom if needed");
    setTimeout(() => {
        const wrapper = tableRef.value?.tableWrapperRef;
        console.log("Table wrapper:", wrapper);
        if (wrapper) {
            // If the content is taller than the visible area, scroll to the bottom.
            if (wrapper.scrollHeight > wrapper.clientHeight) {
                wrapper.scrollTop = wrapper.scrollHeight;
            }
        }
    }, 60);
};
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
                :disabled="isAnyModalOpen"
                @click="togglePolling"
            />
            <!-- Force Refresh Button  -->
            <ForceRefreshButton
                :isPollingLoading="isPollingLoading"
                :isOnline="isOnline"
                :disabled="isAnyModalOpen || isTableLoading"
                @click="forceRefresh"
            />
        </div>

        <TableToolbar title="Defect Types Table" description="">
            <template #left>
                <ToolbarSearch
                    v-model="searchTerm"
                    placeholder="Search defect name or slug..."
                    :disabled="isAnyOperationInProgress"
                    @input="handleSearchChange"
                />

                <FilterResetButton
                    v-if="hasActiveFilters"
                    label="Reset Filters"
                    :disabled="isAnyOperationInProgress"
                    @click="resetAllFilters"
                />
            </template>

            <template #right>
                <CreateButton
                    v-if="isOnline"
                    label="Add Defect Type"
                    title="Add Defect Type"
                    :disabled="isAnyOperationInProgress"
                    @click="handleCreateModalOpen"
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

            <Table ref="tableRef">
                <template #head>
                    <tr class="group/tr table-tr">
                        <TableHeaderCell
                            v-for="header in [
                                {
                                    label: 'Name',
                                    field: 'name',
                                    sortable: true,
                                },
                                {
                                    label: 'Slug',
                                    field: 'slug',
                                    sortable: true,
                                },
                                {
                                    label: 'Description',
                                    field: 'description',
                                    sortable: true,
                                },
                                { label: 'Actions', sortable: false },
                            ]"
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
                        v-for="defectType in displayDefectTypes"
                        v-if="displayDefectTypes.length > 0"
                        :key="defectType.slug || defectType.id || Math.random()"
                        class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                        :class="{ 'opacity-50': isTableLoading }"
                    >
                        <!-- Name Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100 capitalize"
                            >
                                {{ defectType?.name || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Slug Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ defectType?.slug || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Description Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ defectType?.description || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Actions Column -->
                        <TableCell>
                            <EllipsisDropdown
                                :disabled="isAnyOperationInProgress"
                                @click="handleDropdownOpen"
                            >
                                <EditItem
                                    label="Edit Defect Type"
                                    title="Edit Defect Type"
                                    :disabled="isAnyOperationInProgress"
                                    @click="
                                        handleEditModalOpen(defectType.slug)
                                    "
                                />
                                <DeleteItem
                                    label="Delete Defect Type"
                                    title="Delete Defect Type"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleDeleteDefectType(defectType)"
                                />
                            </EllipsisDropdown>
                        </TableCell>
                    </tr>

                    <RowNotFound
                        v-if="
                            displayDefectTypes.length === 0 && !isTableLoading
                        "
                        label="No defect types found"
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

        <!-- Defect Type Create Modal - Teleported to body -->
        <Teleport to="body">
            <DefectTypeCreateModal
                :is-open="isCreateModalOpen"
                @close="handleCreateModalClose"
                @operation="handleDefectTypeOperation"
            />
        </Teleport>

        <!-- Defect Type Edit Modal - Teleported to body -->
        <Teleport to="body">
            <DefectTypeEditModal
                v-if="editDefectTypeSlug"
                :is-open="isEditModalOpen"
                :slug="editDefectTypeSlug"
                @close="handleEditModalClose"
                @operation="handleDefectTypeOperation"
            />
        </Teleport>
    </Tablewrapper>
</template>
