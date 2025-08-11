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
        Schema::create('scans', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->index()->constrained('users')->cascadeOnDelete();
            $table->string('filename')->index();
            $table->string('original_path');
            $table->string('annotated_path');
            $table->boolean('is_defect')->default(false)->index();
            $table->decimal('anomaly_score', 6, 5)->nullable();
            $table->string('anomaly_confidence_level')->nullable()->index();
            $table->decimal('anomaly_inference_time_ms', 8, 3)->nullable();
            $table->decimal('classification_inference_time_ms', 8, 3)->nullable();
            $table->decimal('preprocessing_time_ms', 8, 3)->nullable();
            $table->decimal('postprocessing_time_ms', 8, 3)->nullable();
            $table->decimal('anomaly_threshold', 6, 5)->nullable();
            $table->timestamps();

            // Performance indexes
            $table->index('created_at');
            $table->index('updated_at');

            // Composite indexes for common query patterns
            $table->index(['user_id', 'created_at'], 'scans_user_created_idx');
            $table->index(['user_id', 'is_defect'], 'scans_user_defect_idx');
            $table->index(['user_id', 'created_at', 'is_defect'], 'scans_user_created_defect_idx');

            // Index for date range queries
            $table->index(['created_at', 'is_defect'], 'scans_created_defect_idx');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('scans');
    }
};
