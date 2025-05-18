import type { AllocationResult } from "@/lib/types"

// Placeholder API functions that would connect to your PyTorch backend
// Replace these with actual API calls when backend is ready

export async function fetchBaseAllocation(data: any[]): Promise<AllocationResult[]> {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      // Mock response
      const results = data.map((ship, index) => ({
        shipId: ship.id,
        ATA: ship.ETA + 1,
        ATD: ship.ETD + 1,
        Berth_No: (index % 10) + 1,
      }))
      resolve(results)
    }, 1000)
  })
}

export async function fetchHumanAllocation(data: any[]): Promise<AllocationResult[]> {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      // Mock response with slightly different allocation
      const results = data.map((ship, index) => ({
        shipId: ship.id,
        ATA: ship.ETA + 1.5,
        ATD: ship.ETD + 1.5,
        Berth_No: ((index + 2) % 10) + 1,
      }))
      resolve(results)
    }, 1000)
  })
}

export async function fetchMLAllocation(data: any[]): Promise<AllocationResult[]> {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      // Mock response with ML-based allocation
      const results = data.map((ship, index) => ({
        shipId: ship.id,
        ATA: ship.ETA + 0.5,
        ATD: ship.ETD + 0.5,
        Berth_No: ((index + 5) % 10) + 1,
      }))
      resolve(results)
    }, 1000)
  })
}
