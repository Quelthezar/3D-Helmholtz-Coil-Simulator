from typing import Dict, Optional
import numpy as np


class Coil:
    """
    A class to represent a coil in a Helmholtz coil setup.

    Attributes
    ----------
    coil_width : float
        Width of the coil cross-section in meters. (SQUARE)
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
        constants: Dict[str, float],
        constraints: Dict[str, float],
    ):
        """
        Initialize the Coil object.

        Parameters
        ----------
        coil_width : float
            Width of the coil cross-section in meters. (SQUARE)
        constants : dict
            Dictionary of physical constants used in calculations.
        constraints : dict
            Dictionary of constraints for the coil geometry.
        """
        self.coil_width = coil_width
        self.magnetFieldH = constraints["desiredMagneticFieldB"] / constants["permeabilityFreeSpace"]
        self.currentDensity = constraints["currentDensity"]
        self.constants = constants
        self.constraints = constraints

        # Geometry and electrical properties
        self.deltai = constants["deltai"]
        self.deltao = constants["deltao"]
        self.spool_innerWallThickness = constants["spool_innerWallThickness"]
        self.spool_sideWallThickness = constants["spool_sideWallThickness"]
        self.eta = constants["eta"]

        self.diameter: Optional[float] = None
        self.diameter_inner: Optional[float] = None
        self.diameter_outer: Optional[float] = None
        self.gap: Optional[float] = None
        self.packing_efficiency: Optional[float] = None
        self.xi: Optional[float] = None
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
        self.packing_efficiency = (np.pi * self.deltai ** 2) / (
            4 * self.deltao ** 2
        )

        self.xi = (
            1.43 * self.currentDensity * self.packing_efficiency
        ) / self.magnetFieldH

        if self.coil_width <= self.deltao:
            raise ValueError("Invalid coil width")

        self.diameter = self.xi * (self.coil_width - self.deltao) ** 2
        self.diameter_inner = (
            self.diameter
            - self.coil_width
            - 2 * self.spool_innerWallThickness
        )
        self.diameter_outer = self.diameter + self.coil_width
        self.gap = (
            0.5 * self.diameter
            - self.coil_width
            - 2 * self.spool_sideWallThickness
        )

    def calculate_electrical_properties(self) -> None:
        """
        Calculate the electrical properties of the coil (wire length, resistance, voltage, and power).
        """
        # Effective volume of wire in a single coil
        self.psi = (np.pi / 4) * (self.coil_width - self.deltao) * (
            self.diameter_inner
            + 2 * self.spool_innerWallThickness
            + 2
            * self.deltao
            * np.floor(self.coil_width / self.deltao)
        ) ** 2 - (np.pi / 4) * (self.coil_width - self.deltao) * (
            self.diameter_inner + 2 * self.spool_innerWallThickness
        ) ** 2
        print("psi = ", self.psi)

        # Total length of wire in a Helmholtz pair
        self.lambda_c = 2 * self.psi / (self.deltao ** 2)
        print("lambda_c = ", self.lambda_c)

        # Resistance per unit length of wire
        self.resistance_coil = self.eta * self.lambda_c

        # Current required for the coil (based on chosen current density)
        self.current_coil = (
            np.pi * self.deltai ** 2 / 4
        ) * self.currentDensity
        print("current_coil = ", self.current_coil)

        self.voltage_coil = self.current_coil * self.resistance_coil
        print("voltage_coil = ", self.voltage_coil)

        self.power_coil = self.current_coil**2 * self.resistance_coil
        print("power_coil = ", self.power_coil)

    def calculate_next_larger_coil(self) -> "Coil":
        """
        Calculate the next larger coil's width, gap, and diameters using the current coil's values.

        Returns
        -------
        Coil
            A new Coil object representing the next larger coil.
        """

        # Quartic coefficients based on the current coil's dimensions (from appendix B)
        a4 = 1.25 * self.xi**2
        a3 = -5 * self.xi**2 * self.deltao - 3 * self.xi
        a2 = (
            7.5 * self.xi**2 * self.deltao ** 2
            + 6 * self.xi * self.deltao
            - 2 * self.spool_sideWallThickness * self.xi
            - 4 * self.spool_innerWallThickness * self.xi
            + 2
        )
        a1 = (
            -5 * self.xi**2 * self.deltao ** 3
            - 3 * self.xi * self.deltao ** 2
            + 4
            * (
                self.spool_sideWallThickness
                + 2 * self.spool_innerWallThickness
            )
            * (self.xi * self.deltao + 1)
        )
        a0 = (
            1.25 * self.xi**2 * self.deltao ** 4
            - 2
            * (
                self.spool_sideWallThickness
                + 2 * self.spool_innerWallThickness
            )
            * self.xi
            * self.deltao ** 2
            + 4
            * (
                self.spool_innerWallThickness ** 2
                + self.spool_sideWallThickness ** 2
            )
            - (
                self.gap
                + 2 * self.coil_width
                + 4 * self.spool_sideWallThickness
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
            constants=self.constants,
            constraints=self.constraints,
        )

        # Calculate geometry for the next coil
        next_coil.calculate_geometry()
        next_coil.calculate_electrical_properties()

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
            print(f"\n--- {coil_name} Results ---")

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
    constants: Dict[str, float],
    constraints: Dict[str, float],
    tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Perform a parameter sweep to find the smallest coil width for coil 1 that satisfies the given constraints.

    Parameters
    ----------
    constants : dict
        Dictionary containing relevant constants such as 'deltao', 'spool_innerWallThickness', etc.
    constraints : dict
        Dictionary containing constraints for the coil geometry.
    tolerance : float, optional
        Tolerance for floating-point comparisons, by default 1e-6.

    Returns
    -------
    Optional[float]
        The smallest coil width (in meters) that satisfies the constraints, or None if no valid width is found.
    """
    # Extract constraints
    magnetFieldH = constraints["desiredMagneticFieldB"] / constants["permeabilityFreeSpace"]
    currentDensity = constraints["currentDensity"]

    gap_1_min = constraints["gap_1_min"]
    coil_inner_diameter_min = constraints["diameter_inner_1_min"]

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
        if gap_1 > gap_1_min and diameter_inner_1 > coil_inner_diameter_min - tolerance:
            # Return the first valid coil width that satisfies the constraints
            return coil_width

    # If no valid coil width is found, return None
    return None


# Define constants for Helmholtz coils
constants: Dict[str, float] = {
    "deltai": 0.511 * 10**-3,  # m (minimum diameter)
    "deltao": 0.512 * 10**-3,  # m (maximum diameter)
    "spool_innerWallThickness": 2.5 * 10**-3,  # m
    "spool_sideWallThickness": 2.5 * 10**-3,  # m
    "mu_naught": 4 * np.pi * 10**-7,  # Permeability of free space
    "maxCurrent": 1.25,  # amps
    "eta": 86 / 1000,  # ohms per meter
    "permeabilityFreeSpace": 4 * np.pi * 10**-7,  # Tm/A
}

constraints: Dict[str, float] = {
    "desiredMagneticFieldB": 10e-3,  # T
    "currentDensity": 6e6,  # A/m^2
    "gap_1_min": 20e-3,  # m
    "gap_2_min": 20e-3,  # m
    "gap_3_min": 20e-3,  # m
    "diameter_inner_1_min": 40e-3,  # m
    "diameter_outer_2_max": 70e-3,  # m
    "chordLength_3_max": 0.105,  # m
}

# Example usage for finding the smallest coil width
# Print chosen constants
print("\nChosen maximum magnetic field strength: ", constraints["desiredMagneticFieldB"] * 1000, " mT")
print(f"Chosen current density: ", constraints["currentDensity"], " A/m^2\n")

# Find the smallest coil width for coil 1
smallest_coil_width = find_smallest_coil_width(constants, constraints)

if smallest_coil_width:
    print(f"Calculated smallest coil width: {smallest_coil_width * 1000:.2f} mm\n")

    # Create Coil 1
    coil_1 = Coil(
        coil_width=smallest_coil_width,
        constants=constants,
        constraints=constraints,
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

    # Shorter summary at end, including each coil's coil width, diameter, and gap
    print("\n--- Summary ---")
    print(f"Diameter of wire: {constants['deltai'] * 1000:.2f} mm")
    print(f"Target Magnetic Field Strength: {constraints['desiredMagneticFieldB'] * 1000:.2f} mT")
    print(f"Current Density: {constraints['currentDensity']} A/m^2")
    print(f"\nCoil 1: Width = {coil_1.coil_width * 1000:.2f} mm, Diameter = {coil_1.diameter * 1000:.2f} mm, Gap = {coil_1.gap * 1000:.2f} mm")
    print(f"Coil 2: Width = {coil_2.coil_width* 1000:.2f} mm, Diameter = {coil_2.diameter * 1000:.2f} mm, Gap = {coil_2.gap * 1000:.2f} mm")
    print(f"Coil 3: Width = {coil_3.coil_width * 1000:.2f} mm, Diameter = {coil_3.diameter * 1000:.2f} mm, Gap = {coil_3.gap * 1000:.2f} mm")

    # Calculate the total power consumed by the three coils
    total_power = coil_1.power_coil + coil_2.power_coil + coil_3.power_coil
    print(f"\nTotal power consumed by all coils: {total_power:.2f} W")

    # Print the current through each of the three coils
    print(f"Current through Coil 1: {coil_1.current_coil:.2f} A")
    print(f"Current through Coil 2: {coil_2.current_coil:.2f} A")
    print(f"Current through Coil 3: {coil_3.current_coil:.2f} A")

    # Calculate the total length of wire used in the three coils
    total_wire_length = coil_1.lambda_c + coil_2.lambda_c + coil_3.lambda_c
    print(f"Total length of wire used in all coils: {total_wire_length:.2f} m")
else:
    print("No suitable coil width found in the given range.")
