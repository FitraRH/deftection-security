<?php

namespace Database\Factories;

use App\Models\Product;
use Illuminate\Support\Str;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\Product>
 */
class ProductFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {

        $name = fake()->unique()->company();
        $slug = Str::slug($name);
        $createdAt = fake()->dateTimeBetween('-1 month', 'now');

        // must unique
        while (Product::where('slug', $slug)->exists()) {
            $name = fake()->unique()->company();
            $slug = Str::slug($name);
        }

        return [
            'name' => $name,
            'slug' => $slug,
            'created_at' => $createdAt,
            'updated_at' => $createdAt
        ];
    }
}
