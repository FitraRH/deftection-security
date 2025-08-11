<script setup>
import Sidebar from "./Sidebar/Index.vue";
import MainHeader from "./MainHeader.vue";
</script>

<template>
    <Head>
        <meta
            type="description"
            content="Information about my app"
            head-key="description"
        />
    </Head>

    <div class="min-h-screen">
        <Sidebar :user="$page.props.auth?.user"></Sidebar>

        <div class="flex flex-col flex-1">
            <MainHeader>{{ activeMenu }}</MainHeader>
            <SimpleBar class="h-full">
                <main class="grid grid-cols-1 transition-content main-content">
                    <div
                        class="transition-content mt-4 px-(--margin-x) pb-8 sm:mt-5 lg:mt-6"
                    >
                        <slot />
                    </div>
                </main>
            </SimpleBar>
        </div>
    </div>
</template>
