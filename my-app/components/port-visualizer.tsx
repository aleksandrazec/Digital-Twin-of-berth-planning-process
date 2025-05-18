import type { Ship, AllocationResult } from "@/lib/types"
import { Boat } from "@/components/boat"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface PortVisualizerProps {
  allocation: AllocationResult[]
  ships: Ship[]
}

export function PortVisualizer({ allocation, ships }: PortVisualizerProps) {
  const rows = 5
  const cols = 10
  const berths = Array.from({ length: rows * cols }, (_, i) => i + 1)

  const getShipForBerth = (berthNo: number) => {
    return allocation.find((a) => a.Berth_No === berthNo)
  }

  const getShipById = (id: number) => {
    return ships.find((s) => s.id === id)
  }

  return (
    <div className="bg-[#0077be] p-6 rounded-lg">
      <div className="grid grid-cols-10 gap-2">
        {berths.map((berthNo) => {
          const allocation = getShipForBerth(berthNo)
          const shipId = allocation?.shipId
          const ship = shipId ? getShipById(shipId) : null

          return (
            <TooltipProvider key={berthNo}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className={`
                      h-20 bg-[#a2d5f2] rounded-md flex items-center justify-center
                      relative border border-blue-300
                    `}
                  >
                    <div className="absolute top-1 left-1 text-xs font-bold text-blue-800">{berthNo}</div>
                    {allocation && <Boat shipId={shipId!} shipType={ship?.Type || 1} />}
                  </div>
                </TooltipTrigger>
                {allocation && (
                  <TooltipContent>
                    <div className="text-sm">
                      <p>
                        <strong>Ship ID:</strong> {shipId}
                      </p>
                      <p>
                        <strong>ATA:</strong> {allocation.ATA}
                      </p>
                      <p>
                        <strong>ATD:</strong> {allocation.ATD}
                      </p>
                      <p>
                        <strong>Berth:</strong> {allocation.Berth_No}
                      </p>
                    </div>
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          )
        })}
      </div>
    </div>
  )
}
