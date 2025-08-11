<script setup>
import { ref, computed, watch, nextTick } from "vue";
import {
    Dialog,
    DialogPanel,
    TransitionChild,
    TransitionRoot,
} from "@headlessui/vue";
import { Link, router } from "@inertiajs/vue3";
import { route } from "ziggy-js";

// --- Props & Emits ---
const props = defineProps({
    isOpen: {
        type: Boolean,
        required: true,
    },
});
const emit = defineEmits(["close"]);

// --- State ---
const searchInput = ref(null);
const searchQuery = ref("");
// FIX: We will now track the selected item by its unique `href` instead of its index.
const selectedHref = ref(null);

// --- Page Links ---
const pageGroups = ref([
    {
        name: "General",
        pages: [
            {
                name: "Dashboard",
                href: route("dashboard"),
                icon: "fa-solid fa-home",
            },
            {
                name: "Settings",
                href: "/settings",
                icon: "fa-solid fa-gear",
            },
        ],
    },
    {
        name: "Singular / Batch Image",
        pages: [
            {
                name: "New Scan",
                href: route("scans.create"),
                icon: "fa-solid fa-camera",
            },
            {
                name: "Scan History",
                href: route("scans.index"),
                icon: "fa-solid fa-history",
            },
        ],
    },
    {
        name: "Real-Time Session",
        pages: [
            {
                name: "New Session",
                href: route("sessions.create"),
                icon: "fa-solid fa-satellite-dish",
            },
            {
                name: "Session History",
                href: route("sessions.index"),
                icon: "fa-solid fa-list-alt",
            },
        ],
    },
    {
        name: "Database",
        pages: [
            {
                name: "Products",
                href: route("products.index"),
                icon: "fa-solid fa-box",
            },
            {
                name: "Defect Types",
                href: route("defect_types.index"),
                icon: "fa-solid fa-bug",
            },
            {
                name: "Users",
                href: route("users.index"),
                icon: "fa-solid fa-users",
            },
        ],
    },
]);

// --- Computed Properties ---
const filteredPageGroups = computed(() => {
    if (!searchQuery.value) {
        return pageGroups.value;
    }
    const query = searchQuery.value.toLowerCase();
    return pageGroups.value
        .map((group) => ({
            ...group,
            pages: group.pages.filter((page) =>
                page.name.toLowerCase().includes(query)
            ),
        }))
        .filter((group) => group.pages.length > 0);
});

const flatFilteredPages = computed(() => {
    return filteredPageGroups.value.flatMap((group) => group.pages);
});

// --- Functions ---
const closeModal = () => {
    emit("close");
    setTimeout(() => {
        searchQuery.value = "";
        selectedHref.value = null; // Reset to no selection
    }, 200);
};

// --- Keyboard Navigation ---
const handleKeydown = (event) => {
    const pages = flatFilteredPages.value;
    const totalItems = pages.length;
    if (totalItems === 0) return;

    // Find the current index based on the selected href
    const currentIndex = selectedHref.value
        ? pages.findIndex((p) => p.href === selectedHref.value)
        : -1;

    if (event.key === "ArrowDown") {
        event.preventDefault();
        const nextIndex = (currentIndex + 1) % totalItems;
        selectedHref.value = pages[nextIndex].href;
    } else if (event.key === "ArrowUp") {
        event.preventDefault();
        const prevIndex = (currentIndex - 1 + totalItems) % totalItems;
        selectedHref.value = pages[prevIndex].href;
    } else if (event.key === "Enter") {
        event.preventDefault();
        if (selectedHref.value) {
            router.visit(selectedHref.value);
            closeModal();
        }
    }
};

watch(
    () => props.isOpen,
    (isOpen) => {
        if (isOpen) {
            nextTick(() => {
                searchInput.value?.focus();
            });
        }
    }
);
</script>

<template>
    <TransitionRoot appear :show="isOpen" as="template">
        <Dialog @close="closeModal" class="relative z-50">
            <TransitionChild
                as="template"
                enter="duration-300 ease-out"
                enter-from="opacity-0"
                enter-to="opacity-100"
                leave="duration-200 ease-in"
                leave-from="opacity-100"
                leave-to="opacity-0"
            >
                <div class="fixed inset-0 bg-black/60 backdrop-blur-sm" />
            </TransitionChild>

            <div class="fixed inset-0 overflow-y-auto">
                <div
                    class="flex min-h-full items-start justify-center p-4 pt-[15vh] text-center"
                >
                    <TransitionChild
                        as="template"
                        enter="duration-300 ease-out"
                        enter-from="opacity-0 scale-95"
                        enter-to="opacity-100 scale-100"
                        leave="duration-200 ease-in"
                        leave-from="opacity-100 scale-100"
                        leave-to="opacity-0 scale-95"
                    >
                        <DialogPanel
                            class="w-full max-w-xl transform overflow-hidden rounded-lg bg-white p-4 text-left align-middle shadow-xl transition-all dark:bg-dark-800"
                            @keydown="handleKeydown"
                        >
                            <!-- Search Input -->
                            <div class="relative">
                                <input
                                    ref="searchInput"
                                    v-model="searchQuery"
                                    type="text"
                                    placeholder="Search for a page..."
                                    class="w-full rounded-md border-gray-300 bg-gray-100 py-3 pl-10 pr-4 text-gray-900 focus:border-primary-500 focus:ring-primary-500 dark:border-dark-600 dark:bg-dark-700 dark:text-dark-100"
                                />
                                <div
                                    class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3"
                                >
                                    <font-awesome-icon
                                        icon="fa-solid fa-search"
                                        class="text-gray-400"
                                    />
                                </div>
                            </div>

                            <!-- Results -->
                            <div class="mt-4 max-h-[50vh] overflow-y-auto">
                                <div
                                    v-if="flatFilteredPages.length === 0"
                                    class="p-4 text-center"
                                >
                                    No pages found.
                                </div>
                                <div
                                    v-for="group in filteredPageGroups"
                                    :key="group.name"
                                >
                                    <h3
                                        class="px-2 pt-4 pb-2 text-xs font-semibold uppercase"
                                    >
                                        {{ group.name }}
                                    </h3>
                                    <ul @mouseleave="selectedHref = null">
                                        <li
                                            v-for="page in group.pages"
                                            :key="page.href"
                                        >
                                            <Link
                                                :href="page.href"
                                                @click="closeModal"
                                                @mouseenter="
                                                    selectedHref = page.href
                                                "
                                                class="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-gray-700 transition-colors dark:text-dark-200"
                                                :class="{
                                                    'bg-primary-500/10 text-primary-600 dark:bg-primary-500/20 dark:text-primary-300':
                                                        selectedHref ===
                                                        page.href,
                                                }"
                                            >
                                                <font-awesome-icon
                                                    :icon="page.icon"
                                                    class="w-5"
                                                />
                                                <span>{{ page.name }}</span>
                                            </Link>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </DialogPanel>
                    </TransitionChild>
                </div>
            </div>
        </Dialog>
    </TransitionRoot>
</template>
