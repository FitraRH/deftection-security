<script setup>
import { ref, defineExpose } from "vue";

const tableWrapperRef = ref(null);

defineExpose({
    tableWrapperRef,
});
</script>
<template>
    <div
        ref="tableWrapperRef"
        class="table-wrapper min-w-full overflow-x-auto grow"
    >
        <table class="table w-full text-left rtl:text-right is-hoverable">
            <thead class="group/thead table-thead">
                <slot name="head"></slot>
            </thead>
            <tbody class="group/tbody table-tbody">
                <slot name="body"></slot>
            </tbody>
        </table>
    </div>
</template>
