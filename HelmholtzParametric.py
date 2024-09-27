from typing import Dict, Optional
import numpy as np


class Coil:
    """
    A class to represent a coil in a Helmholtz coil setup.

    Attributes
    ----------
    coil_width : float
        Width of the coil in meters.
    magnetFieldH : float
        Magnetic field strength in A/m.
    currentDensity : float
        Current density in A/m^2.
    constants : dict
        Dictionary containing physical constants like 'deltao', 'spool_innerWallThickness', etc.
    diameter : Optional[float]
        Calculated nominal diameter of the coil in meters.
    diameter_inner : Optional[float]
        Inner diameter of the coil in meters.
    diameter_outer : Optional[float]
        Outer diameter of the coil in meters.
    gap : Optional[float]
        Gap between individual coils in a pair in meters.
    lambda_c : Optional[float]
        Length of the wire in meters.
    resistance_coil : Optional[float]
        Resistance of the coil in Ohms.
    current_coil : Optional[float]
        Current through the coil in Amps.
    voltage_coil : Optional[float]
        Voltage required for the coil in Volts.
    power_coil : Optional[float]
        Power consumed by the coil in Watts.
    """

    def __init__(
        self,
        coil_width: float,
        magnetFieldH: float,
        currentDensity: float,
        constants: Dict[str, float],
    ):
        """
        Initialize the Coil object.

        Parameters
        ----------
        coil_width : float
            Width of the coil in meters.
        magnetFieldH : float
            Magnetic field strength in A/m.
        currentDensity : float
            Current density in A/m^2.
        constants : dict
            Dictionary of physical constants used in calculations.
        """
        self.coil_width = coil_width
        self.magnetFieldH = magnetFieldH
        self.currentDensity = currentDensity
        self.constants = constants

        # Geometry and electrical properties
        self.diameter: Optional[float] = None
        self.diameter_inner: Optional[float] = None
        self.diameter_outer: Optional[float] = None
        self.gap: Optional[float] = None
        self.psi: Optional[float] = None
        self.lambda_c: Optional[float] = None
        self.resistance_coil: Optional[float] = None
        self.current_coil: Optional[float] = None
        self.voltage_coil: Optional[float] = None
        self.power_coil: Optional[float] = None

    def calculate_geometry(self) -> None:
        """
        Calculate the geometric properties of the coil (diameter, inner diameter, outer diameter, and gap).
        """
        packing_efficiency = (np.pi * self.constants["deltai"] ** 2) / (
            4 * self.constants["deltao"] ** 2
        )

        xi = (1.43 * self.currentDensity * packing_efficiency) / self.magnetFieldH
        if self.coil_width <= self.constants["deltao"]:
            raise ValueError("Invalid coil width")

        self.diameter = xi * (self.coil_width - self.constants["deltao"]) ** 2
        self.diameter_inner = (
            self.diameter
            - self.coil_width
            - 2 * self.constants["spool_innerWallThickness"]
        )
        self.diameter_outer = self.diameter + self.coil_width
        self.gap = (
            0.5 * self.diameter
            - self.coil_width
            - 2 * self.constants["spool_sideWallThickness"]
        )

    def calculate_electrical_properties(self) -> None:
        """
        Calculate the electrical properties of the coil (wire length, resistance, voltage, and power).
        """
        self.psi = (np.pi / 4) * (self.coil_width - self.constants["deltao"]) * (
            self.diameter_inner
            + 2 * self.constants["spool_innerWallThickness"]
            + 2
            * self.constants["deltao"]
            * np.floor(self.coil_width / self.constants["deltao"])
        ) ** 2 - (np.pi / 4) * (self.coil_width - self.constants["deltao"]) * (
            self.diameter_inner + 2 * self.constants["spool_innerWallThickness"]
        ) ** 2
        print("psi", self.psi)

        self.lambda_c = 2 * self.psi / self.constants["deltao"] ** 2
        print("lambda_c", self.lambda_c)
        self.resistance_coil = self.constants["eta"] * self.lambda_c

        self.current_coil = (
            np.pi * self.constants["deltai"] ** 2 / 4
        ) * self.currentDensity
        self.voltage_coil = self.current_coil * self.resistance_coil
        self.power_coil = self.current_coil**2 * self.resistance_coil

    def calculate_next_larger_coil(self) -> "Coil":
        """
        Calculate the next larger coil's width, gap, and diameters using the current coil's values.

        Returns
        -------
        Coil
            A new Coil object representing the next larger coil.
        """
        # Calculate xi for current setup
        packing_efficiency = (np.pi * self.constants["deltai"] ** 2) / (
            4 * self.constants["deltao"] ** 2
        )
        xi = (1.43 * self.currentDensity * packing_efficiency) / self.magnetFieldH

        # Quartic coefficients based on the current coil's dimensions
        a4 = 1.25 * xi**2
        a3 = -5 * xi**2 * self.constants["deltao"] - 3 * xi
        a2 = (
            7.5 * xi**2 * self.constants["deltao"] ** 2
            + 6 * xi * self.constants["deltao"]
            - 2 * self.constants["spool_sideWallThickness"] * xi
            - 4 * self.constants["spool_innerWallThickness"] * xi
            + 2
        )
        a1 = (
            -5 * xi**2 * self.constants["deltao"] ** 3
            - 3 * xi * self.constants["deltao"] ** 2
            + 4
            * (
                self.constants["spool_sideWallThickness"]
                + 2 * self.constants["spool_innerWallThickness"]
            )
            * (xi * self.constants["deltao"] + 1)
        )
        a0 = (
            1.25 * xi**2 * self.constants["deltao"] ** 4
            - 2
            * (
                self.constants["spool_sideWallThickness"]
                + 2 * self.constants["spool_innerWallThickness"]
            )
            * xi
            * self.constants["deltao"] ** 2
            + 4
            * (
                self.constants["spool_innerWallThickness"] ** 2
                + self.constants["spool_sideWallThickness"] ** 2
            )
            - (
                self.gap
                + 2 * self.coil_width
                + 4 * self.constants["spool_sideWallThickness"]
            )
            ** 2
            - self.diameter_outer**2
        )

        # Solve the quartic equation to get the next coil width
        roots = np.roots([a4, a3, a2, a1, a0])

        # Filter out complex roots (keep only real roots)
        real_roots = roots[np.isreal(roots)].real

        # Filter for positive real roots (ignore negative roots)
        positive_real_roots = real_roots[real_roots > 0]

        if len(positive_real_roots) == 0:
            raise ValueError(
                "No valid positive real root found for the next coil width."
            )

        # Choose the smallest positive root as the next coil width
        next_coil_width = np.min(positive_real_roots)

        # Create the next larger coil based on this width
        next_coil = Coil(
            coil_width=next_coil_width,
            magnetFieldH=self.magnetFieldH,
            currentDensity=self.currentDensity,
            constants=self.constants,
        )

        # Calculate geometry for the next coil
        next_coil.calculate_geometry()

        return next_coil

    def display_results(self, coil_name: Optional[str] = None) -> None:
        """
        Display the geometry and electrical properties of the coil.

        Parameters
        ----------
        coil_name : Optional[str]
            The name of the coil (e.g., "coil_1", "coil_2") to include in the output. Default is None.
        """
        self.calculate_geometry()

        # Display coil name if provided
        if coil_name:
            print(f"--- {coil_name} Results ---")

        print(f"Coil Width: {self.coil_width} m or {self.coil_width * 1000} mm")
        print(
            f"Inner Diameter: {self.diameter_inner} m or {self.diameter_inner * 1000} mm"
        )
        print(
            f"Outer Diameter: {self.diameter_outer} m or {self.diameter_outer * 1000} mm"
        )
        print(f"Gap between coils: {self.gap} m or {self.gap * 1000} mm")

        self.calculate_electrical_properties()
        print(f"Wire Length: {self.lambda_c} m")
        print(f"Voltage Required: {self.voltage_coil} V")
        print(f"Power Consumed: {self.power_coil} W")


def find_smallest_coil_width(
    magnetFieldH: float,
    currentDensity: float,
    constants: Dict[str, float],
    tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Perform a parameter sweep to find the smallest coil width for coil 1 that satisfies the given constraints.

    Parameters
    ----------
    magnetFieldH : float
        Magnetic field strength in A/m.
    currentDensity : float
        Current density in A/m^2.
    constants : dict
        Dictionary containing relevant constants such as 'deltao', 'spool_innerWallThickness', etc.
    tolerance : float, optional
        Tolerance for floating-point comparisons, by default 1e-6.

    Returns
    -------
    Optional[float]
        The smallest coil width (in meters) that satisfies the constraints, or None if no valid width is found.
    """
    # Define the range of possible coil widths to search through
    coilWidth_1_range = np.linspace(0.0001, 0.1, 1000000)

    # Perform the sweep over possible coil widths
    for coil_width in coilWidth_1_range:
        # Calculate xi, the factor that scales with current density and packing efficiency
        packing_efficiency = (np.pi * constants["deltai"] ** 2) / (
            4 * constants["deltao"] ** 2
        )
        xi = (1.43 * currentDensity * packing_efficiency) / magnetFieldH

        # Skip values where the coil width is smaller than or equal to the wire's outer diameter
        if coil_width <= constants["deltao"]:
            continue

        # Calculate the diameter and inner diameter of the coil
        diameter_1 = xi * (coil_width - constants["deltao"]) ** 2
        diameter_inner_1 = (
            diameter_1 - coil_width - 2 * constants["spool_innerWallThickness"]
        )

        # Calculate the gap between the coils
        gap_1 = 0.5 * diameter_1 - coil_width - 2 * constants["spool_sideWallThickness"]

        # Check the constraints: gap must be larger than 0.02 m and the inner diameter must satisfy the tolerance
        if gap_1 > 0.02 and diameter_inner_1 > 40 * 10**-3 - tolerance:
            # Return the first valid coil width that satisfies the constraints
            return coil_width

    # If no valid coil width is found, return None
    return None


# Define constants for Helmholtz coils
constants = {
    "deltai": 0.511 * 10**-3,  # m (minimum diameter)
    "deltao": 0.512 * 10**-3,  # m (maximum diameter)
    "spool_innerWallThickness": 2.5 * 10**-3,  # m
    "spool_sideWallThickness": 2.5 * 10**-3,  # m
    "mu_naught": 4 * np.pi * 10**-7,  # Permeability of free space
    "maxCurrent": 1.25,  # amps
    "eta": 86 / 1000,  # ohms per meter
}

# Example usage for finding the smallest coil width
desiredMagneticFieldB = 10E-3  # T
permeabiltyFreeSpace = 4 * np.pi * 10**-7  # Tm/A
magnetFieldH = desiredMagneticFieldB/permeabiltyFreeSpace  # A/m
currentDensity = 6E6  # A/m^2

# Constraints for coils
gap_1_min = 20E-3  # m
gap_2_min = 20E-3  # m
gap_3_min = 20E-3  # m

diameter_inner_1_min = 40E-3  # m
diameter_outer_2_max = 70E-3  # m

chordLength_3_max = 0.105  # m

# Print chosen constants
print(f"\nChosen maximum magnetic field strength: {desiredMagneticFieldB*1000} mT")
print(f"Chosen current density: {currentDensity} A/m^2\n")

# Find the smallest coil width for coil 1
smallest_coil_width: float = find_smallest_coil_width(
    magnetFieldH, currentDensity, constants
)

if smallest_coil_width:
    print(f"Calculated smallest coil width: {smallest_coil_width * 1000:.2f} mm\n")

    # Create Coil 1
    coil_1 = Coil(
        coil_width=smallest_coil_width,
        magnetFieldH=79580,
        currentDensity=2.5 * 10**6,
        constants=constants,
    )

    # Calculate geometry for Coil 1
    coil_1.calculate_geometry()

    # Now calculate Coil 2 based on Coil 1
    coil_2 = coil_1.calculate_next_larger_coil()

    # Now calculate Coil 3 based on Coil 2
    coil_3 = coil_2.calculate_next_larger_coil()

    # Display results for Coil 1, Coil 2, and Coil 3
    coil_1.display_results(coil_name="Coil 1")
    coil_2.display_results(coil_name="Coil 2")
    coil_3.display_results(coil_name="Coil 3")
else:
    print("No suitable coil width found in the given range.")
