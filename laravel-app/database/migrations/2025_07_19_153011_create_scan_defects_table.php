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
        Schema::create('scan_defects', function (Blueprint $table) {
            $table->id();
            $table->foreignId('scan_id')->constrained('scans')->cascadeOnDelete();
            $table->string('label')->index();
            $table->decimal('confidence_score', 6, 5)->nullable();
            $table->string('severity_level')->nullable()->index();
            $table->decimal('area_percentage', 5, 2)->nullable();
            $table->json('box_location')->nullable();
            $table->timestamps();

            // Performance indexes
            $table->index('created_at');
            $table->index('updated_at');

            // Composite indexes for efficient queries
            $table->index(['scan_id', 'label'], 'scan_defects_scan_label_idx');
            $table->index(['label', 'severity_level'], 'scan_defects_label_severity_idx');

            // Index for filtering by confidence score
            $table->index('confidence_score');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('scan_defects');
    }
};
