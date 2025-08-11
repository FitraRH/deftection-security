<script setup>
import { ref, onMounted, computed, onUnmounted } from "vue";
import { usePage } from "@inertiajs/vue3";
import { useUiStore } from "../stores/useUiStore";
import { useThemeStore } from "../stores/useThemeStore";
import { storeToRefs } from "pinia";

// Import shared components
import SearchModal from "./Modals/SearchModal.vue";

const uiStore = useUiStore();

const handleSidebarToggle = () => {
    uiStore.toggleSidebar();
};

const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);

const toggleTheme = () => themeStore.toggleTheme();

// Search modal logic
const isSearchModalOpen = ref(false);
const openSearchModal = () => {
    isSearchModalOpen.value = true;
    console.log("Search modal opened");
};

// --- Keyboard Shortcut Listener ---
const handleKeyPress = (event) => {
    // Open search with '/' key, but not when typing in an input field
    if (
        event.key === "/" &&
        event.target.tagName !== "INPUT" &&
        event.target.tagName !== "TEXTAREA"
    ) {
        event.preventDefault();
        openSearchModal();
    }
};

// Check for saved theme preference on mount
onMounted(() => {
    themeStore.initializeTheme();
    window.addEventListener("keydown", handleKeyPress);
});

onUnmounted(() => {
    window.removeEventListener("keydown", handleKeyPress);
});
</script>

<template>
    <header
        class="app-header transition-content sticky top-0 z-20 flex h-[65px] items-center gap-2 border-b border-gray-200 bg-white/80 px-(--margin-x) backdrop-blur-sm backdrop-saturate-150 dark:border-dark-600 max-sm:justify-between dark:bg-dark-900/80"
    >
        <div class="xl:hidden contents">
            <button
                @click="handleSidebarToggle"
                class="flex flex-col justify-center space-y-1.5 rtl:mr-0.5 ltr:ml-0.5 outline-hidden focus:outline-hidden size-7 text-primary-600 dark:text-primary-400 cursor-pointer sidebar-toggle-btn"
            >
                <span></span><span></span><span></span>
            </button>
        </div>
        <div class="flex flex-1 justify-end items-center gap-2">
            <div class="flex-1">
                <button
                    @click="openSearchModal"
                    class="flex cursor-pointer items-center gap-4 outline-hidden max-sm:hidden"
                >
                    <div class="flex items-center gap-2">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            aria-hidden="true"
                            data-slot="icon"
                            class="size-5"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
                            ></path></svg
                        ><span>Search here...</span>
                    </div>
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="22"
                        height="20"
                        aria-hidden="true"
                    >
                        <path
                            fill="none"
                            stroke="currentColor"
                            d="M3.5.5h12c1.7 0 3 1.3 3 3v13c0 1.7-1.3 3-3 3h-12c-1.7 0-3-1.3-3-3v-13c0-1.7 1.3-3 3-3z"
                            opacity="0.4"
                        ></path>
                        <path
                            fill="currentColor"
                            d="M11.8 6L8 15.1h-.9L10.8 6h1z"
                        ></path>
                    </svg></button
                ><button
                    class="btn-base btn shrink-0 p-0 hover:bg-gray-300/20 focus:bg-gray-300/20 text-gray-700 active:bg-gray-300/25 dark:text-dark-200 dark:hover:bg-dark-300/10 dark:focus:bg-dark-300/10 dark:active:bg-dark-300/20 relative size-9 rounded-full sm:hidden"
                    type="button"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        fill="none"
                        viewBox="0 0 24 24"
                        class="size-6 text-gray-900 dark:text-dark-100"
                    >
                        <path
                            fill="currentColor"
                            d="M10.5 19a8.5 8.5 0 1 0 0-17 8.5 8.5 0 0 0 0 17Z"
                            opacity="0.3"
                        ></path>
                        <path
                            fill="currentColor"
                            d="M20.92 22a1.07 1.07 0 0 1-.752-.308l-2.857-2.859a1.086 1.086 0 0 1 0-1.522 1.084 1.084 0 0 1 1.52 0l2.858 2.86a1.086 1.086 0 0 1 0 1.521c-.215.2-.492.308-.768.308Z"
                        ></path>
                    </svg>
                </button>
            </div>

            <!-- Dark/Light Toggle Button -->
            <button
                @click="toggleTheme"
                class="p-2 rounded-lg text-dark-200 hover:text-primary-600 transition-all duration-200 cursor-pointer"
            >
                <font-awesome-icon
                    :icon="isDark ? 'fa-solid fa-sun' : 'fa-solid fa-moon'"
                    class="text-lg"
                />
            </button>
        </div>
    </header>

    <SearchModal
        :is-open="isSearchModalOpen"
        @close="isSearchModalOpen = false"
    />
</template>
