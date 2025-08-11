<?php

namespace Database\Factories;

use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\RealtimeSession>
 */
class RealtimeSessionFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        $sessionStart = $this->faker->dateTimeBetween('-1 month', 'now');
        $sessionStatus = $this->faker->randomElement(['completed', 'aborted', 'active', 'paused']);

        // If session is completed, set end time and calculate duration
        $sessionEnd = null;
        $durationSeconds = null;

        if ($sessionStatus === 'completed') {
            // Ensure we have a valid end time after start time
            $maxEndTime = max($sessionStart, new \DateTime('-1 day'));
            $sessionEnd = $this->faker->dateTimeBetween($sessionStart, 'now');
            $durationSeconds = $sessionEnd->getTimestamp() - $sessionStart->getTimestamp();
        } elseif ($sessionStatus === 'aborted') {
            // Aborted sessions have a short duration
            $maxEndTime = clone $sessionStart;
            $maxEndTime->modify('+1 hour');
            $endTimeLimit = min(new \DateTime(), $maxEndTime);

            // Only create end time if it's after start time
            if ($endTimeLimit > $sessionStart) {
                $sessionEnd = $this->faker->dateTimeBetween($sessionStart, $endTimeLimit);
                $durationSeconds = $sessionEnd->getTimestamp() - $sessionStart->getTimestamp();
            }
        }

        $totalFrames = $this->faker->numberBetween(100, 5000);
        $defectCount = $this->faker->numberBetween(0, $totalFrames * 0.3); // Max 30% defects
        $goodCount = $totalFrames - $defectCount;

        $defectRate = $totalFrames > 0 ? round(($defectCount / $totalFrames) * 100, 2) : 0;
        $goodRate = $totalFrames > 0 ? round(($goodCount / $totalFrames) * 100, 2) : 0;

        return [
            'user_id' => User::factory(),
            'session_status' => $sessionStatus,
            'session_start' => $sessionStart,
            'session_end' => $sessionEnd,
            'duration_seconds' => $durationSeconds,
            'camera_location' => $this->faker->randomElement(['Station A', 'Station B', 'Station C', 'Production Line 1', 'Production Line 2']),
            'total_frames_processed' => $totalFrames,
            'throughput_fps' => $this->faker->randomFloat(3, 15.0, 60.0),
            'defect_count' => $defectCount,
            'defect_rate' => $defectRate,
            'good_count' => $goodCount,
            'good_rate' => $goodRate,
            'avg_processing_time' => $this->faker->randomFloat(3, 50.0, 200.0),
            'max_processing_time' => $this->faker->randomFloat(3, 200.0, 500.0),
            'min_processing_time' => $this->faker->randomFloat(3, 20.0, 50.0),
            'avg_anomaly_score' => $this->faker->randomFloat(5, 0.1, 0.9),
            'max_anomaly_score' => $this->faker->randomFloat(5, 0.8, 1.0),
            'min_anomaly_score' => $this->faker->randomFloat(5, 0.0, 0.2),
            'avg_classification_confidence' => $this->faker->randomFloat(5, 0.7, 0.95),
            'max_classification_confidence' => $this->faker->randomFloat(5, 0.95, 1.0),
            'min_classification_confidence' => $this->faker->randomFloat(5, 0.5, 0.7),
            'defect_type_distribution' => [
                'scratch' => $this->faker->numberBetween(0, 20),
                'dent' => $this->faker->numberBetween(0, 15),
                'corrosion' => $this->faker->numberBetween(0, 10),
                'crack' => $this->faker->numberBetween(0, 8),
                'discoloration' => $this->faker->numberBetween(0, 12),
                'missing_part' => $this->faker->numberBetween(0, 5),
            ],
            'severity_level_distribution' => [
                'low' => $this->faker->numberBetween(0, 50),
                'medium' => $this->faker->numberBetween(0, 30),
                'high' => $this->faker->numberBetween(0, 10),
            ],
            'created_at' => $sessionStart,
            'updated_at' => $sessionEnd ?? $sessionStart,
        ];
    }
}
