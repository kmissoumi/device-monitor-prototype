#!/usr/bin/env python3
"""
Sauce Labs Real Device State Monitor

Continuously monitors the state of private real devices and tracks
time spent in each state with comprehensive analytics and reporting.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
import shutil

import click
import requests
from requests.auth import HTTPBasicAuth
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
#import keyboard

console = Console()

class DeviceStatus(Enum):
    AVAILABLE = "AVAILABLE"
    IN_USE = "IN_USE"
    CLEANING = "CLEANING"
    REBOOTING = "REBOOTING"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_string(cls, value: str):
        """Safely convert string to DeviceStatus enum."""
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN

@dataclass
class StateTransition:
    """Represents a single state transition for a device."""
    state: str
    start: str  # ISO timestamp
    end: Optional[str] = None  # ISO timestamp
    duration_seconds: Optional[float] = None
    in_use_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DeviceState:
    """Tracks the complete state history and statistics for a device."""
    descriptor: str
    current_state: str
    current_state_start: str  # ISO timestamp
    state_history: List[StateTransition] = field(default_factory=list)
    cumulative_stats: Dict[str, float] = field(default_factory=dict)
    last_seen: str = ""
    in_use_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "descriptor": self.descriptor,
            "current_state": self.current_state,
            "current_state_start": self.current_state_start,
            "state_history": [t.to_dict() for t in self.state_history],
            "cumulative_stats": self.cumulative_stats,
            "last_seen": self.last_seen,
            "in_use_by": self.in_use_by
        }

class SauceLabsAPI:
    """API client for Sauce Labs RDC endpoints."""
    
    def __init__(self, username: str, access_key: str, region: str = "us-west-1", debug: bool = False):
        self.username = username
        self.access_key = access_key
        self.auth = HTTPBasicAuth(username, access_key)
        self.base_url = f"https://api.{region}.saucelabs.com/rdc/v2"
        self.debug = debug
        
    def get_device_status(self) -> Optional[List[Dict[str, Any]]]:
        """Get device status from V2 API."""
        url = f"{self.base_url}/devices/status"
        params = {"privateOnly": "true"}
        
        try:
            response = requests.get(url, auth=self.auth, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("devices", [])
        except requests.RequestException as e:
            logging.error(f"API Error: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None

class DeviceMonitor:
    """Main device monitoring class."""
    
    def __init__(self, api: SauceLabsAPI, data_file: str, backup_interval: int = 300):
        self.api = api
        self.data_file = Path(data_file)
        self.backup_interval = backup_interval
        self.devices: Dict[str, DeviceState] = {}
        self.session_start = datetime.now(timezone.utc).isoformat()
        self.running = False
        self.last_backup = time.time()
        
        # Display state
        self.current_view = "overview"  # overview, history, stats
        self.time_filter = "all"  # all, 1h, 24h, 7d
        self.display_refresh_interval = 2.0  # seconds
        
        # Load existing data if available
        self.load_data()
    
    def load_data(self):
        """Load existing monitoring data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                self.session_start = data.get("session_start", self.session_start)
                
                for descriptor, device_data in data.get("devices", {}).items():
                    # Reconstruct StateTransition objects
                    history = []
                    for t_data in device_data.get("state_history", []):
                        transition = StateTransition(
                            state=t_data["state"],
                            start=t_data["start"],
                            end=t_data.get("end"),
                            duration_seconds=t_data.get("duration_seconds"),
                            in_use_by=t_data.get("in_use_by", [])
                        )
                        history.append(transition)
                    
                    device_state = DeviceState(
                        descriptor=descriptor,
                        current_state=device_data["current_state"],
                        current_state_start=device_data["current_state_start"],
                        state_history=history,
                        cumulative_stats=device_data.get("cumulative_stats", {}),
                        last_seen=device_data.get("last_seen", ""),
                        in_use_by=device_data.get("in_use_by", [])
                    )
                    self.devices[descriptor] = device_state
                
                logging.info(f"Loaded data for {len(self.devices)} devices from {self.data_file}")
            except Exception as e:
                logging.error(f"Error loading data: {e}")
                logging.info("Starting with fresh data")
    
    def save_data(self, backup: bool = False):
        """Save current monitoring data to file."""
        data = {
            "session_start": self.session_start,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "devices": {desc: device.to_dict() for desc, device in self.devices.items()}
        }
        
        filename = self.data_file
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.data_file.with_suffix(f".backup_{timestamp}.json")
        
        try:
            # Write to temp file first, then rename for atomicity
            temp_file = filename.with_suffix(filename.suffix + ".tmp")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(filename)
            
            if backup:
                logging.info(f"Backup saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
    
    def update_device_state(self, descriptor: str, new_state: str, in_use_by: List[str]):
        """Update device state and handle transitions."""
        now = datetime.now(timezone.utc).isoformat()
        
        if descriptor not in self.devices:
            # New device discovered
            logging.info(f"New device discovered: {descriptor} in state {new_state}")
            self.devices[descriptor] = DeviceState(
                descriptor=descriptor,
                current_state=new_state,
                current_state_start=now,
                last_seen=now,
                in_use_by=in_use_by
            )
            return
        
        device = self.devices[descriptor]
        device.last_seen = now
        device.in_use_by = in_use_by
        
        # Check for state transition
        if device.current_state != new_state:
            # Finalize previous state
            duration = (datetime.fromisoformat(now.replace('Z', '+00:00')) - 
                       datetime.fromisoformat(device.current_state_start.replace('Z', '+00:00'))).total_seconds()
            
            # Create transition record
            transition = StateTransition(
                state=device.current_state,
                start=device.current_state_start,
                end=now,
                duration_seconds=duration,
                in_use_by=device.in_use_by
            )
            device.state_history.append(transition)
            
            # Update cumulative stats
            if device.current_state not in device.cumulative_stats:
                device.cumulative_stats[device.current_state] = 0
            device.cumulative_stats[device.current_state] += duration
            
            logging.info(f"Device {descriptor}: {device.current_state} -> {new_state} (duration: {duration:.1f}s)")
            
            # Start new state
            device.current_state = new_state
            device.current_state_start = now
    
    def get_time_filtered_stats(self, device: DeviceState, time_filter: str) -> Dict[str, float]:
        """Calculate statistics for a specific time range."""
        if time_filter == "all":
            return device.cumulative_stats.copy()
        
        now = datetime.now(timezone.utc)
        if time_filter == "1h":
            cutoff = now - timedelta(hours=1)
        elif time_filter == "24h":
            cutoff = now - timedelta(hours=24)
        elif time_filter == "7d":
            cutoff = now - timedelta(days=7)
        else:
            return device.cumulative_stats.copy()
        
        stats = {}
        
        # Process completed transitions in time range
        for transition in device.state_history:
            if not transition.end:
                continue
            
            start_time = datetime.fromisoformat(transition.start.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(transition.end.replace('Z', '+00:00'))
            
            # Skip if entirely before cutoff
            if end_time < cutoff:
                continue
            
            # Calculate overlap with time range
            effective_start = max(start_time, cutoff)
            effective_duration = (end_time - effective_start).total_seconds()
            
            if effective_duration > 0:
                if transition.state not in stats:
                    stats[transition.state] = 0
                stats[transition.state] += effective_duration
        
        # Handle current state if it overlaps with time range
        current_start = datetime.fromisoformat(device.current_state_start.replace('Z', '+00:00'))
        if current_start < now:
            effective_start = max(current_start, cutoff)
            if effective_start < now:
                current_duration = (now - effective_start).total_seconds()
                if device.current_state not in stats:
                    stats[device.current_state] = 0
                stats[device.current_state] += current_duration
        
        return stats
    
    def create_overview_table(self) -> Table:
        """Create the main overview table."""
        table = Table(title=f"Device Status Overview ({self.time_filter})", box=box.ROUNDED)
        
        table.add_column("Device", style="cyan", no_wrap=True)
        table.add_column("Current State", style="bold")
        table.add_column("Time in State", style="yellow")
        table.add_column("In Use By", style="dim")
        table.add_column("Available %", justify="right", style="green")
        table.add_column("In Use %", justify="right", style="blue")
        table.add_column("Cleaning %", justify="right", style="cyan")
        table.add_column("Rebooting %", justify="right", style="magenta")
        table.add_column("Maintenance %", justify="right", style="red")
        table.add_column("Offline %", justify="right", style="dim")
        
        now = datetime.now(timezone.utc)
        
        for descriptor in sorted(self.devices.keys()):
            device = self.devices[descriptor]
            
            # Calculate time in current state
            current_start = datetime.fromisoformat(device.current_state_start.replace('Z', '+00:00'))
            time_in_current = (now - current_start).total_seconds()
            
            # Format time in current state
            if time_in_current < 60:
                time_str = f"{time_in_current:.0f}s"
            elif time_in_current < 3600:
                time_str = f"{time_in_current/60:.1f}m"
            else:
                time_str = f"{time_in_current/3600:.1f}h"
            
            # Get filtered stats and calculate percentages
            stats = self.get_time_filtered_stats(device, self.time_filter)
            total_time = sum(stats.values())
            
            if total_time > 0:
                available_pct = (stats.get("AVAILABLE", 0) / total_time) * 100
                in_use_pct = (stats.get("IN_USE", 0) / total_time) * 100
                cleaning_pct = (stats.get("CLEANING", 0) / total_time) * 100
                rebooting_pct = (stats.get("REBOOTING", 0) / total_time) * 100
                maintenance_pct = (stats.get("MAINTENANCE", 0) / total_time) * 100
                offline_pct = (stats.get("OFFLINE", 0) / total_time) * 100
            else:
                available_pct = in_use_pct = cleaning_pct = rebooting_pct = maintenance_pct = offline_pct = 0
            
            # Color current state
            state_color = self._get_state_color(device.current_state)
            state_text = f"[{state_color}]{device.current_state}[/{state_color}]"
            
            # Format users
            users_text = ", ".join(device.in_use_by) if device.in_use_by else "-"
            if len(users_text) > 20:
                users_text = users_text[:17] + "..."
            
            table.add_row(
                descriptor,
                state_text,
                time_str,
                users_text,
                f"{available_pct:.1f}%",
                f"{in_use_pct:.1f}%",
                f"{cleaning_pct:.1f}%",
                f"{rebooting_pct:.1f}%",
                f"{maintenance_pct:.1f}%",
                f"{offline_pct:.1f}%"
            )
        
        return table
    
    def create_history_table(self) -> Table:
        """Create the recent history table."""
        table = Table(title="Recent State Changes (Last 20)", box=box.ROUNDED)
        
        table.add_column("Time", style="dim")
        table.add_column("Device", style="cyan", no_wrap=True)
        table.add_column("From", style="yellow")
        table.add_column("To", style="green")
        table.add_column("Duration", style="blue")
        table.add_column("Users", style="dim")
        
        # Collect all recent transitions
        all_transitions = []
        for device in self.devices.values():
            for transition in device.state_history:
                if transition.end:
                    all_transitions.append((device.descriptor, transition))
        
        # Sort by end time and take last 20
        all_transitions.sort(key=lambda x: x[1].end, reverse=True)
        
        for descriptor, transition in all_transitions[:20]:
            device = self.devices[descriptor]
            
            # Find what state came after this transition
            next_state = device.current_state
            for i, t in enumerate(device.state_history):
                if t == transition and i < len(device.state_history) - 1:
                    next_state = device.state_history[i + 1].state
                    break
            
            # Format time
            end_time = datetime.fromisoformat(transition.end.replace('Z', '+00:00'))
            time_str = end_time.strftime("%H:%M:%S")
            
            # Format duration
            if transition.duration_seconds < 60:
                duration_str = f"{transition.duration_seconds:.0f}s"
            elif transition.duration_seconds < 3600:
                duration_str = f"{transition.duration_seconds/60:.1f}m"
            else:
                duration_str = f"{transition.duration_seconds/3600:.1f}h"
            
            # Format users
            users_text = ", ".join(transition.in_use_by) if transition.in_use_by else "-"
            if len(users_text) > 15:
                users_text = users_text[:12] + "..."
            
            from_color = self._get_state_color(transition.state)
            to_color = self._get_state_color(next_state)
            
            table.add_row(
                time_str,
                descriptor,
                f"[{from_color}]{transition.state}[/{from_color}]",
                f"[{to_color}]{next_state}[/{to_color}]",
                duration_str,
                users_text
            )
        
        return table
    
    def create_stats_panel(self) -> Panel:
        """Create statistics panel."""
        total_devices = len(self.devices)
        
        if total_devices == 0:
            return Panel("No devices being monitored", title="Statistics")
        
        # Count current states
        state_counts = {}
        state_times = {"AVAILABLE": 0, "IN_USE": 0, "CLEANING": 0, "REBOOTING": 0, "MAINTENANCE": 0, "OFFLINE": 0, "UNKNOWN": 0}
        
        for device in self.devices.values():
            if device.current_state not in state_counts:
                state_counts[device.current_state] = 0
            state_counts[device.current_state] += 1
            
            stats = self.get_time_filtered_stats(device, self.time_filter)
            for state, time_val in stats.items():
                if state in state_times:
                    state_times[state] += time_val
                else:
                    state_times[state] = state_times.get(state, 0) + time_val
        
        total_time = sum(state_times.values())
        
        stats_text = f"Total Devices: {total_devices}\n\n"
        stats_text += "Current States:\n"
        for state, count in sorted(state_counts.items()):
            color = self._get_state_color(state)
            stats_text += f"  [{color}]{state}[/{color}]: {count}\n"
        
        if total_time > 0:
            stats_text += f"\nOverall Utilization ({self.time_filter}):\n"
            for state, time_val in sorted(state_times.items()):
                if time_val > 0:
                    color = self._get_state_color(state)
                    percentage = (time_val / total_time) * 100
                    stats_text += f"  [{color}]{state}[/{color}]: {percentage:.1f}%\n"
        
        return Panel(stats_text, title="Statistics", border_style="blue")
    
    def _get_state_color(self, state: str) -> str:
        """Get color for device state."""
        colors = {
            "AVAILABLE": "green",
            "IN_USE": "yellow",
            "CLEANING": "blue",
            "REBOOTING": "blue",
            "MAINTENANCE": "red",
            "OFFLINE": "red",
            "UNKNOWN": "magenta"
        }
        return colors.get(state, "white")
    
    def create_display(self) -> Layout:
        """Create the main display layout."""
        layout = Layout()
        
        # Header
        time_str = datetime.now().strftime("%H:%M:%S")
        running_time = datetime.now(timezone.utc) - datetime.fromisoformat(self.session_start.replace('Z', '+00:00'))
        running_str = str(running_time).split('.')[0]  # Remove microseconds
        
        header_text = f"Sauce Labs Device Monitor | {time_str} | Running: {running_str} | View: {self.current_view} | Filter: {self.time_filter}"
        header = Panel(Align.center(header_text), style="bold blue")
        
        # Controls
        controls = Panel(
            "Controls: [q]+Enter to quit | [1][2][3]+Enter for views | [h][d][w][a]+Enter for time filters | [s]+Enter to save | [b]+Enter to backup",
            style="dim"
        )
        
        layout.split_column(
            Layout(header, size=3),
            Layout(name="main"),
            Layout(controls, size=3)
        )
        
        # Main content based on current view
        if self.current_view == "overview":
            layout["main"].update(self.create_overview_table())
        elif self.current_view == "history":
            layout["main"].update(self.create_history_table())
        elif self.current_view == "stats":
            layout["main"].update(self.create_stats_panel())
        
        return layout
    
    def poll_devices(self):
        """Poll device status from API."""
        devices_data = self.api.get_device_status()
        if devices_data is None:
            logging.warning("Failed to fetch device status")
            return
        
        # Process each device
        current_descriptors = set()
        for device_data in devices_data:
            descriptor = device_data["descriptor"]
            state = device_data.get("status") or device_data.get("state", "UNKNOWN")
            in_use_by = []
            
            if device_data.get("inUseBy"):
                in_use_by = [
                    user.get("username", "unknown") if isinstance(user, dict) else str(user)
                    for user in device_data["inUseBy"]
                ]
            
            current_descriptors.add(descriptor)
            self.update_device_state(descriptor, state, in_use_by)
        
        # Handle devices that disappeared
        for descriptor in list(self.devices.keys()):
            if descriptor not in current_descriptors:
                device = self.devices[descriptor]
                # Only mark as offline if we haven't seen it for a while
                last_seen = datetime.fromisoformat(device.last_seen.replace('Z', '+00:00'))
                if (datetime.now(timezone.utc) - last_seen).total_seconds() > 300:  # 5 minutes
                    if device.current_state != "OFFLINE":
                        self.update_device_state(descriptor, "OFFLINE", [])
    
    def run_monitoring(self, interval: int, silent: bool = False):
        """Run the main monitoring loop."""
        self.running = True
        
        if silent:
            # Silent mode - just log and save data
            logging.info("Starting silent monitoring mode")
            try:
                while self.running:
                    self.poll_devices()
                    
                    # Periodic backup
                    if time.time() - self.last_backup > self.backup_interval:
                        self.save_data(backup=True)
                        self.last_backup = time.time()
                    
                    # Save data every minute
                    self.save_data()
                    
                    time.sleep(interval)
            except KeyboardInterrupt:
                logging.info("Stopping monitoring")
            finally:
                self.save_data()
                self.running = False
        else:
            # Interactive mode with rich display
            def handle_input():
                """Handle keyboard input in a separate thread."""
                while self.running:
                    try:
                        key = input()  # This will block until user presses Enter
                        if key.lower() == 'q':
                            self.running = False
                        elif key == '1':
                            self.current_view = "overview"
                        elif key == '2':
                            self.current_view = "history"
                        elif key == '3':
                            self.current_view = "stats"
                        elif key.lower() == 'h':
                            self.time_filter = "1h"
                        elif key.lower() == 'd':
                            self.time_filter = "24h"
                        elif key.lower() == 'w':
                            self.time_filter = "7d"
                        elif key.lower() == 'a':
                            self.time_filter = "all"
                        elif key.lower() == 's':
                            self.save_data()
                        elif key.lower() == 'b':
                            self.save_data(backup=True)
                    except (EOFError, KeyboardInterrupt):
                        self.running = False
                        break
            
            # Start input handler in thread
            input_thread = threading.Thread(target=handle_input, daemon=True)
            input_thread.start()
            
            # Main display loop
            with Live(self.create_display(), refresh_per_second=1/self.display_refresh_interval) as live:
                last_poll = 0
                while self.running:
                    current_time = time.time()
                    
                    # Poll API at specified interval
                    if current_time - last_poll >= interval:
                        self.poll_devices()
                        last_poll = current_time
                        
                        # Periodic backup
                        if current_time - self.last_backup > self.backup_interval:
                            self.save_data(backup=True)
                            self.last_backup = current_time
                        
                        # Save data
                        self.save_data()
                    
                    # Update display
                    live.update(self.create_display())
                    time.sleep(0.1)
            
            # Final save
            self.save_data()

@click.command()
@click.option('--interval', default=5, help='Polling interval in seconds')
@click.option('--region', envvar='SAUCE_REGION', default='us-west-1', 
              help='Sauce Labs region (or set SAUCE_REGION env var)')
@click.option('--data-file', default='device_monitoring.json', 
              help='Data persistence file')
@click.option('--log-dir', default='.', 
              help='Directory for log files')
@click.option('--backup-interval', default=300, 
              help='Backup interval in seconds')
@click.option('--resume/--fresh-start', default=True, 
              help='Resume from existing data or start fresh')
@click.option('--silent', is_flag=True, 
              help='Run in silent mode (no interactive display)')
@click.option('--debug', is_flag=True, 
              help='Enable debug logging')
def main(interval, region, data_file, log_dir, backup_interval, resume, silent, debug):
    """
    Sauce Labs Real Device State Monitor
    
    Continuously monitors private device states and tracks time in each state.
    """
    
    # Setup logging
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging to file
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / 'device_monitor.log'),
            logging.StreamHandler() if silent else logging.NullHandler()
        ]
    )
    
    # Get credentials
    username = os.getenv('SAUCE_USERNAME')
    access_key = os.getenv('SAUCE_ACCESS_KEY')
    
    if not username or not access_key:
        console.print("[red]Error: SAUCE_USERNAME and SAUCE_ACCESS_KEY environment variables must be set[/red]")
        sys.exit(1)
    
    # Validate data file path
    data_path = Path(data_file)
    if not resume and data_path.exists():
        if not silent:
            from rich.prompt import Confirm
            if not Confirm.ask(f"Data file {data_file} exists. Overwrite?"):
                console.print("Aborted")
                sys.exit(0)
        data_path.unlink()
    
    try:
        # Initialize API and monitor
        api = SauceLabsAPI(username, access_key, region, debug)
        monitor = DeviceMonitor(api, data_file, backup_interval)
        
        if not silent:
            console.print("[green]Starting device monitoring...[/green]")
            console.print(f"[dim]Polling interval: {interval}s | Region: {region} | Data file: {data_file}[/dim]")
            console.print("[dim]Press 'q' to quit, '1-3' to change views, 'h/d/w/a' for time filters[/dim]")
        
        logging.info(f"Starting monitoring - interval: {interval}s, region: {region}")
        
        # Run monitoring
        monitor.run_monitoring(interval, silent)
        
    except KeyboardInterrupt:
        if not silent:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")
        logging.info("Monitoring stopped by user")
    except Exception as e:
        error_msg = f"Error: {e}"
        if not silent:
            console.print(f"[red]{error_msg}[/red]")
        logging.error(error_msg, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
