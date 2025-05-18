import type { Ship, AllocationResult } from "@/lib/types"
import { PortVisualizer } from "@/components/port-visualizer"

interface ComparisonViewProps {
  title: string
  allocation: AllocationResult[]
  ships: Ship[]
}

export function ComparisonView({ title, allocation, ships }: ComparisonViewProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">{title}</h3>
      <PortVisualizer allocation={allocation} ships={ships} />

      {}
      <div className="mt-4 bg-white p-4 rounded-lg shadow-sm">
        <h4 className="font-medium mb-2">Allocation Statistics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 p-3 rounded-md">
            <p className="text-sm text-gray-500">Total Ships</p>
            <p className="text-xl font-semibold">{allocation.length}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded-md">
            <p className="text-sm text-gray-500">Berths Used</p>
            <p className="text-xl font-semibold">{new Set(allocation.map((a) => a.Berth_No)).size}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded-md">
            <p className="text-sm text-gray-500">Avg. Delay</p>
            <p className="text-xl font-semibold">
              {allocation.length > 0
                ? (
                    allocation.reduce((sum, a) => {
                      const ship = ships.find((s) => s.id === a.shipId)
                      return sum + (a.ATA - (ship?.ETA || 0))
                    }, 0) / allocation.length
                  ).toFixed(1)
                : "0.0"}
            </p>
          </div>
          <div className="bg-gray-50 p-3 rounded-md">
            <p className="text-sm text-gray-500">Utilization</p>
            <p className="text-xl font-semibold">
              {allocation.length > 0
                ? `${Math.round((new Set(allocation.map((a) => a.Berth_No)).size / 50) * 100)}%`
                : "0%"}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
