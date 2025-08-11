<?php

namespace App\Policies;

use App\Models\User;
use App\Models\RealtimeSession;
use Illuminate\Auth\Access\HandlesAuthorization;

class RealtimeSessionPolicy
{
    use HandlesAuthorization;

    /**
     * Determine whether the user can view any realtime sessions.
     */
    public function viewAny(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can view the realtime session.
     */
    public function view(User $user, RealtimeSession $realtimeSession): bool
    {
        return $user->role === 'admin' || $realtimeSession->user_id === $user->id;
    }

    /**
     * Determine whether the user can create realtime sessions.
     */
    public function create(User $user): bool
    {
        return true; // All authenticated users can create sessions
    }

    /**
     * Determine whether the user can update the realtime session.
     */
    public function update(User $user, RealtimeSession $realtimeSession): bool
    {
        return $user->role === 'admin' || $realtimeSession->user_id === $user->id;
    }

    /**
     * Determine whether the user can delete the realtime session.
     */
    public function delete(User $user, RealtimeSession $realtimeSession): bool
    {
        return $user->role === 'admin' || $realtimeSession->user_id === $user->id;
    }

    public function deleteAny(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can filter by user (admin only).
     */
    public function filterByUser(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can generate a report for a specific session.
     */
    public function generateReport(User $user, RealtimeSession $realtimeSession): bool
    {
        // User can generate report for their own sessions, admin can generate for any session
        return $user->role === 'admin' || $realtimeSession->user_id === $user->id;
    }

    /**
     * Determine whether the user can generate batch reports for any user.
     */
    public function generateAnyReport(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can generate batch reports.
     * Users can generate batch reports for their own data, admins for any data.
     */
    public function generateBatchReport(User $user): bool
    {
        return true;
    }
}
