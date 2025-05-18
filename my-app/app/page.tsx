"use client"

import { useState, useEffect } from "react"
import FixedInputTable from "@/components/fixed-input-table"
import { DynamicInputTable } from "@/components/dynamic-input-table"
import { PortVisualizer } from "@/components/port-visualizer"
import { ComparisonView } from "@/components/comparison-view"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { fetchBaseAllocation, fetchHumanAllocation, fetchMLAllocation } from "@/lib/api"
import type { Ship, AllocationResult } from "@/lib/types"

export default function BerthAllocationSystem() {
  // Sample initial data
  const [fixedData, setFixedData] = useState<Ship[]>([])
useEffect(() => {
  fetch("/api/ships")
    .then(res => res.json())
    .then(data => setFixedData(data))
}, [])

useEffect(() => {
  console.log(fixedData)
}, [fixedData])

const [dynamicData, setDynamicData] = useState<Record<string, any>>({})

useEffect(() => {
  fetch("./api/ships2")
    .then(res => res.json())
    .then(data => {
      const dyn: Record<string, any> = {}
      data.forEach((ship: any) => {
        dyn[ship.CALL_SIGN] = {
          Weather: Number(ship.WEATHER_IMPACT_PCT) || 0,
          Congestion: Number(ship.CONGESTION_IMPACT_PCT) || 0,
          Effectiveness: Number(ship.EFFECTIVENESS_SCORE) || 0,
          Reliability: Number(ship.RELIABILITY_SCORE) || 0,
          Work_Environment: Number(ship.WORK_ENV_SCORE) || 0,
        }
      })
      setDynamicData(dyn)
    })
}, [])

  // Allocation results
  const [baseAllocation, setBaseAllocation] = useState<AllocationResult[]>([])
  const [humanAllocation, setHumanAllocation] = useState<AllocationResult[]>([])
  const [mlAllocation, setMLAllocation] = useState<AllocationResult[]>([])
  const [loading, setLoading] = useState(false)

  // Handle dynamic data changes
const handleDynamicDataChange = (shipId: string, field: string, value: number) => {
  setDynamicData((prev) => ({
    ...prev,
    [shipId]: {
      ...prev[shipId],
      [field]: value,
    },
  }))
}

  // Submit data to backend
  const handleSubmit = async () => {
    setLoading(true)
    try {
      // Combine fixed and dynamic data
      const combinedData = fixedData.map((ship) => ({
        ...ship,
        ...dynamicData[ship.id],
      }))

      // Fetch allocations from different algorithms
      const baseResults = await fetchBaseAllocation(combinedData)
      const humanResults = await fetchHumanAllocation(combinedData)
      const mlResults = await fetchMLAllocation(combinedData)

      setBaseAllocation(baseResults)
      setHumanAllocation(humanResults)
      setMLAllocation(mlResults)
    } catch (error) {
      console.error("Error fetching allocations:", error)
      // For demo purposes, set mock data if API fails
      setMockAllocationData()
    } finally {
      setLoading(false)
    }
  }

  // Set mock allocation data for demonstration
  const setMockAllocationData = () => {
    const mockBase = fixedData.map((ship, index) => ({
      shipId: ship.id,
      ATA: ship.ETA + 1,
      ATD: ship.ETD + 1,
      Berth_No: (index % 10) + 1,
    }))

    const mockHuman = fixedData.map((ship, index) => ({
      shipId: ship.id,
      ATA: ship.ETA + 1.5,
      ATD: ship.ETD + 1.5,
      Berth_No: ((index + 2) % 10) + 1,
    }))

    const mockML = fixedData.map((ship, index) => ({
      shipId: ship.id,
      ATA: ship.ETA + 0.5,
      ATD: ship.ETD + 0.5,
      Berth_No: ((index + 5) % 10) + 1,
    }))

    setBaseAllocation(mockBase)
    setHumanAllocation(mockHuman)
    setMLAllocation(mockML)
  }

  // Load mock data on initial render for demonstration
  useEffect(() => {
    setMockAllocationData()
  }, [])
console.log("shipIds", Object.keys(dynamicData))
  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8 text-center">Berth Allocation System</h1>

      {/* Top Section - Input Tables */}
      <div className="mb-8">
        {/* <h2 className="text-xl font-semibold mb-4">Ship Information</h2> */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-medium mb-2">Fixed Ship Data</h3>
            <FixedInputTable data={fixedData} />
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-medium mb-2">Editable Parameters</h3>
           <DynamicInputTable
              data={dynamicData}
              shipIds={Object.keys(dynamicData)}
              onChange={handleDynamicDataChange}

            />
          </div>
        </div>
        <div className="mt-6 flex justify-center">
          <Button onClick={handleSubmit} disabled={loading} className="px-8 py-2">
            {loading ? "Processing..." : "Submit for Allocation"}
          </Button>
        </div>
      </div>

      {/* Middle Section - Base Algorithm Port Layout */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Base Algorithm Allocation</h2>
        <PortVisualizer allocation={baseAllocation} ships={fixedData} />
      </div>

      {/* Bottom Section - Comparison Views */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Allocation Comparisons</h2>
        <Tabs defaultValue="human">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="human">Human-Simulated Allocation</TabsTrigger>
            <TabsTrigger value="ml">Machine Learning Allocation</TabsTrigger>
          </TabsList>
          <TabsContent value="human">
            <ComparisonView title="Human-Simulated Allocation" allocation={humanAllocation} ships={fixedData} />
          </TabsContent>
          <TabsContent value="ml">
            <ComparisonView title="Machine Learning Allocation" allocation={mlAllocation} ships={fixedData} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
