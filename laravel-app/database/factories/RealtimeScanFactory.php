<?php

namespace Database\Factories;

use App\Models\RealtimeSession;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\RealtimeScan>
 */
class RealtimeScanFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        $format = $this->faker->randomElement(['jpg', 'png', 'jpeg']);
        $capturedAt = $this->faker->dateTimeBetween('-1 month', 'now');
        $isDefect = $this->faker->boolean(25); // 25% chance of defect

        return [
            'realtime_session_id' => RealtimeSession::factory(),
            'filename' => 'frame_' . $this->faker->unique()->numberBetween(1000, 9999) . '.' . $format,
            'captured_at' => $capturedAt,
            'annotated_path' => $isDefect ? $this->faker->imageUrl(640, 480, null, true, null, false, $format) : null,
            'is_defect' => $isDefect,
            'anomaly_score' => $this->faker->randomFloat(5, 0, 1),
            'anomaly_confidence_level' => $this->faker->randomElement(['low', 'medium', 'high']),
            'anomaly_inference_time_ms' => $this->faker->randomFloat(3, 10.0, 150.0),
            'classification_inference_time_ms' => $this->faker->randomFloat(3, 5.0, 100.0),
            'preprocessing_time_ms' => $this->faker->randomFloat(3, 2.0, 50.0),
            'postprocessing_time_ms' => $this->faker->randomFloat(3, 1.0, 30.0),
            'anomaly_threshold' => $this->faker->randomFloat(5, 0.3, 0.8),
            'created_at' => $capturedAt,
            'updated_at' => $capturedAt,
        ];
    }
}
