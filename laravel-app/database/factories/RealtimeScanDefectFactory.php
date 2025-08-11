<?php

namespace Database\Factories;

use App\Models\RealtimeScan;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\RealtimeScanDefect>
 */
class RealtimeScanDefectFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        $defectTypes = [
            'scratch',
            'dent',
            'corrosion',
            'crack',
            'discoloration',
            'missing_part'
        ];

        $severityLevels = ['low', 'medium', 'high'];

        return [
            'realtime_scan_id' => RealtimeScan::factory(),
            'label' => $this->faker->randomElement($defectTypes),
            'confidence_score' => $this->faker->randomFloat(5, 0.5, 1.0),
            'severity_level' => $this->faker->randomElement($severityLevels),
            'area_percentage' => $this->faker->randomFloat(2, 0.5, 25.0),
            'box_location' => [
                'x' => $this->faker->numberBetween(10, 90),
                'y' => $this->faker->numberBetween(10, 90),
                'width' => $this->faker->numberBetween(5, 50),
                'height' => $this->faker->numberBetween(5, 50),
            ],
            'created_at' => now(),
            'updated_at' => now(),
        ];
    }
}
