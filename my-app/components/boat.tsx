"use client"

import { useEffect, useRef } from "react"
import boatIcon from "./ui/boat_svg.svg"

interface BoatProps {
  shipId: number
  shipType: number
}

export function Boat({ shipId, shipType }: BoatProps) {
  const boatRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!boatRef.current) return

    const boat = boatRef.current
   
    let animationId: number
    const start = performance.now()

    const amplitude = 2
    const frequency = 0.002

    const animate = (time: number) => {
      const elapsed = time - start
      const offset = Math.sin(elapsed * frequency) * amplitude
      boat.style.transform = `translateX(${offset}px)`
      animationId = requestAnimationFrame(animate)
    }

    animationId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animationId)
    }
  }, [])

  return (
    <div
      ref={boatRef}
      className="relative w-fit h-fit transition-transform"
    >
      <img
        src={boatIcon.src || boatIcon}
        alt="Boat Icon"
        className="w-10 h-10"
      />
      <span className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 text-xs text-black font-bold">
        {shipId}
      </span>
    </div>
  )
}
