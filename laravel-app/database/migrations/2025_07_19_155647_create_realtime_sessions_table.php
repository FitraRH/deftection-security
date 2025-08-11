<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {

        Schema::create('realtime_sessions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained('users')->cascadeOnDelete();
            $table->string('session_status')->default('aborted')->index();
            $table->timestamp('session_start')->index();
            $table->timestamp('session_end')->nullable();
            $table->integer('duration_seconds')->nullable()->index();
            $table->string('camera_location')->nullable();
            $table->integer('total_frames_processed')->default(0)->index();
            $table->decimal('throughput_fps', 6, 3)->default(0.000);
            $table->integer('defect_count')->default(0);
            $table->decimal('defect_rate', 5, 2)->default(0.00)->index();
            $table->integer('good_count')->default(0);
            $table->decimal('good_rate', 5, 2)->default(0.00);
            $table->decimal('avg_processing_time', 8, 3)->nullable();
            $table->decimal('max_processing_time', 8, 3)->nullable();
            $table->decimal('min_processing_time', 8, 3)->nullable();
            $table->decimal('avg_anomaly_score', 6, 5)->nullable();
            $table->decimal('max_anomaly_score', 6, 5)->nullable();
            $table->decimal('min_anomaly_score', 6, 5)->nullable();
            $table->decimal('avg_classification_confidence', 6, 5)->nullable();
            $table->decimal('max_classification_confidence', 6, 5)->nullable();
            $table->decimal('min_classification_confidence', 6, 5)->nullable();
            $table->json('defect_type_distribution')->nullable();
            $table->json('severity_level_distribution')->nullable();
            $table->timestamps();

            // Composite indexes for optimal query performance
            $table->index(['user_id', 'session_status']);
            $table->index(['user_id', 'session_start']);
            $table->index(['session_status', 'session_start']);
            $table->index(['user_id', 'session_status', 'session_start']);
            $table->index(['session_start', 'session_status']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('realtime_sessions');
    }
};
