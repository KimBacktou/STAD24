#!/usr/bin/env python3

from mavsdk import System
import asyncio

async def connect_to_px4():
    drone = System()  # Creating a drone instance
    await drone.connect(system_address="udp://:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break
    
    return drone

async def set_sdlog_mode(system):
    param_plugin = system.param

    # Set SDLOG_MODE to -1 (Disable logging)
    await param_plugin.set_param_int('SDLOG_MODE', -1)
    print("SDLOG_MODE set to -1")

async def main():
    system = await connect_to_px4()
    await set_sdlog_mode(system)

# Run the main function
asyncio.run(main())

