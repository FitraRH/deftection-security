<script setup>
import { ref, computed } from "vue";
import { usePage, useForm } from "@inertiajs/vue3"; // Import useForm
import { successToast } from "@/utils/swal";

// --- Fetch User Data Directly ---
const page = usePage();
const user = computed(() => page.props.auth.user);

// --- Form Management ---
// NEW: Use Inertia's useForm for the profile update
const profileForm = useForm({
    name: user.value?.name || "",
});

const passwordForm = useForm({
    current_password: "",
    new_password: "",
    new_password_confirmation: "",
});

// --- State Management ---
const isEditingProfile = ref(false);

// --- Event Handlers ---
const startEditingProfile = () => {
    // Sync the form's name with the current user's name when editing starts
    profileForm.name = user.value?.name || "";
    isEditingProfile.value = true;
};

const cancelEditProfile = () => {
    isEditingProfile.value = false;
    profileForm.reset("name"); // Reset any validation errors if cancelled
};

// MODIFIED: This now sends a PATCH request to the backend
const saveProfile = () => {
    profileForm.patch(route("settings.account_name.update"), {
        onSuccess: () => {
            successToast("Profile updated successfully!");
            isEditingProfile.value = false;
        },
        onError: () => {
            // Handle errors, e.g., show a notification
            console.error("Failed to update profile.");
        },
    });
};

const submitPassword = () => {
    passwordForm.patch(route("settings.account_password.update"), {
        onSuccess: () => {
            successToast("Password changed successfully! Please log in again.");
            passwordForm.reset();
        },
        onError: (errors) => {
            // You can loop through errors and display them
            console.error("Password change failed:", errors);
        },
    });
};
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <!-- Card 1: User Settings -->
        <div class="flex flex-col">
            <h2
                class="mb-1 flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                User Settings
            </h2>
            <p class="mb-6 text-xs text-gray-400 dark:text-dark-300">
                Manage your account settings and preferences.
            </p>

            <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
                <!-- Full Name Section -->
                <div>
                    <label
                        for="fullName"
                        class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                        >Full Name</label
                    >
                    <!-- Display Mode -->
                    <div
                        v-if="!isEditingProfile"
                        class="flex items-center gap-4"
                    >
                        <p
                            class="form-input-base form-input bg-gray-50 dark:bg-dark-700 border-transparent"
                        >
                            {{ user?.name }}
                        </p>
                        <button
                            @click="startEditingProfile"
                            class="btn-base btn btn-sm this:primary bg-this text-white hover:bg-this-darker"
                        >
                            Edit
                        </button>
                    </div>
                    <!-- Editing Mode -->
                    <div v-else class="flex items-center gap-2">
                        <input
                            v-model="profileForm.name"
                            type="text"
                            id="fullName"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                        <button
                            @click="saveProfile"
                            :disabled="profileForm.processing"
                            class="btn-base btn btn-sm this:success bg-this text-white hover:bg-this-darker"
                        >
                            Save
                        </button>
                        <button
                            @click="cancelEditProfile"
                            class="btn-base btn btn-sm bg-gray-200 text-gray-800 hover:bg-gray-300 dark:bg-dark-600 dark:text-dark-200 dark:hover:bg-dark-500"
                        >
                            Cancel
                        </button>
                    </div>
                    <p
                        v-if="profileForm.errors.name"
                        class="text-xs text-red-500 mt-1"
                    >
                        {{ profileForm.errors.name }}
                    </p>
                </div>
            </div>
        </div>

        <div class="mt-6 mb-8 h-px bg-gray-200 dark:bg-dark-500"></div>

        <!-- Card 2: Change Password -->
        <div class="flex flex-col gap-4">
            <h2
                class="flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                Change Password
            </h2>

            <form @submit.prevent="submitPassword">
                <div class="grid grid-cols-1 gap-6 sm:grid-cols-3">
                    <!-- Current Password -->
                    <div>
                        <label
                            for="currentPassword"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >Current Password</label
                        >
                        <input
                            v-model="passwordForm.current_password"
                            type="password"
                            id="currentPassword"
                            placeholder="Enter your current password"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                        <p
                            v-if="passwordForm.errors.current_password"
                            class="text-xs text-red-500 mt-1"
                        >
                            {{ passwordForm.errors.current_password }}
                        </p>
                    </div>
                    <!-- New Password -->
                    <div>
                        <label
                            for="newPassword"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >New Password</label
                        >
                        <input
                            v-model="passwordForm.new_password"
                            type="password"
                            id="newPassword"
                            placeholder="Enter your new password"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                    </div>
                    <!-- Confirm Password -->
                    <div>
                        <label
                            for="confirmPassword"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >Confirm Password</label
                        >
                        <input
                            v-model="passwordForm.new_password_confirmation"
                            type="password"
                            id="confirmPassword"
                            placeholder="Confirm your new password"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                        <p
                            v-if="passwordForm.errors.new_password"
                            class="text-xs text-red-500 mt-1"
                        >
                            {{ passwordForm.errors.new_password }}
                        </p>
                    </div>
                </div>
                <div class="mt-6 flex justify-end">
                    <button
                        type="submit"
                        class="btn-base btn this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                        :disabled="passwordForm.processing"
                    >
                        Change Password
                    </button>
                </div>
            </form>
        </div>
    </div>
</template>
