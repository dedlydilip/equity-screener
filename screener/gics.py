"""GICS classification helpers.

Maps a GICS **Sub-Industry** (the finest level Wikipedia exposes for the S&P 500)
to its **Industry Group** — the mid-tier used as the *most-neutral* grouping in
the normalization hierarchy (industry_group -> sector -> cross-sectional).

This is the standard GICS structure (~2023). It is intentionally *conservative*:
sub-industries that were recently reclassified or are ambiguous are omitted, so
``industry_group_for`` returns ``None`` and those names fall back to Sector-level
neutralization (which the normalizer reports as a fallback %). Extend the maps
below as needed — accuracy over coverage.
"""

from __future__ import annotations

# Industry Group -> member Sub-Industries (GICS ~2023).
INDUSTRY_GROUP_MEMBERS: dict[str, list[str]] = {
    # Energy
    "Energy": [
        "Oil & Gas Drilling", "Oil & Gas Equipment & Services", "Integrated Oil & Gas",
        "Oil & Gas Exploration & Production", "Oil & Gas Refining & Marketing",
        "Oil & Gas Storage & Transportation", "Coal & Consumable Fuels",
    ],
    # Materials
    "Materials": [
        "Commodity Chemicals", "Diversified Chemicals", "Fertilizers & Agricultural Chemicals",
        "Industrial Gases", "Specialty Chemicals", "Construction Materials",
        "Metal & Glass Containers", "Paper & Plastic Packaging Products & Materials",
        "Aluminum", "Diversified Metals & Mining", "Copper", "Gold",
        "Precious Metals & Minerals", "Silver", "Steel", "Forest Products", "Paper Products",
        "Metal, Glass & Plastic Containers",
    ],
    # Industrials
    "Capital Goods": [
        "Aerospace & Defense", "Building Products", "Construction & Engineering",
        "Electrical Components & Equipment", "Heavy Electrical Equipment",
        "Industrial Conglomerates", "Construction Machinery & Heavy Transportation Equipment",
        "Agricultural & Farm Machinery", "Industrial Machinery & Supplies & Components",
        "Trading Companies & Distributors",
    ],
    "Commercial & Professional Services": [
        "Commercial Printing", "Environmental & Facilities Services", "Office Services & Supplies",
        "Diversified Support Services", "Security & Alarm Services",
        "Human Resource & Employment Services", "Research & Consulting Services",
    ],
    "Transportation": [
        "Air Freight & Logistics", "Passenger Airlines", "Airlines", "Marine Transportation",
        "Rail Transportation", "Cargo Ground Transportation", "Passenger Ground Transportation",
    ],
    # Consumer Discretionary
    "Automobiles & Components": [
        "Automotive Parts & Equipment", "Tires & Rubber", "Automobile Manufacturers",
        "Motorcycle Manufacturers",
    ],
    "Consumer Durables & Apparel": [
        "Consumer Electronics", "Household Appliances", "Housewares & Specialties",
        "Homebuilding", "Leisure Products", "Apparel, Accessories & Luxury Goods",
        "Footwear", "Textiles",
    ],
    "Consumer Services": [
        "Casinos & Gaming", "Hotels, Resorts & Cruise Lines", "Leisure Facilities",
        "Restaurants", "Education Services", "Specialized Consumer Services",
    ],
    "Consumer Discretionary Distribution & Retail": [
        "Distributors", "Broadline Retail", "Apparel Retail", "Computer & Electronics Retail",
        "Home Improvement Retail", "Specialty Stores", "Automotive Retail", "Homefurnishing Retail",
        "Other Specialty Retail",
    ],
    # Consumer Staples
    "Consumer Staples Distribution & Retail": [
        "Drug Retail", "Food Distributors", "Food Retail", "Consumer Staples Merchandise Retail",
    ],
    "Food, Beverage & Tobacco": [
        "Brewers", "Distillers & Vintners", "Soft Drinks & Non-alcoholic Beverages",
        "Agricultural Products & Services", "Packaged Foods & Meats", "Tobacco",
    ],
    "Household & Personal Products": ["Household Products", "Personal Care Products"],
    # Health Care
    "Health Care Equipment & Services": [
        "Health Care Equipment", "Health Care Supplies", "Health Care Distributors",
        "Health Care Services", "Health Care Facilities", "Managed Health Care",
        "Health Care Technology",
    ],
    "Pharmaceuticals, Biotechnology & Life Sciences": [
        "Biotechnology", "Pharmaceuticals", "Life Sciences Tools & Services",
    ],
    # Financials
    "Banks": ["Diversified Banks", "Regional Banks"],
    "Financial Services": [
        "Diversified Financial Services", "Multi-Sector Holdings", "Specialized Finance",
        "Consumer Finance", "Transaction & Payment Processing Services",
        "Asset Management & Custody Banks", "Investment Banking & Brokerage",
        "Financial Exchanges & Data", "Commercial & Residential Mortgage Finance",
    ],
    "Insurance": [
        "Insurance Brokers", "Life & Health Insurance", "Multi-line Insurance",
        "Property & Casualty Insurance", "Reinsurance",
    ],
    # Information Technology
    "Software & Services": [
        "Application Software", "Systems Software", "Internet Services & Infrastructure",
        "IT Consulting & Other Services",
    ],
    "Technology Hardware & Equipment": [
        "Communications Equipment", "Technology Hardware, Storage & Peripherals",
        "Electronic Equipment & Instruments", "Electronic Components",
        "Electronic Manufacturing Services", "Technology Distributors",
    ],
    "Semiconductors & Semiconductor Equipment": [
        "Semiconductors", "Semiconductor Materials & Equipment",
    ],
    # Communication Services
    "Telecommunication Services": [
        "Alternative Carriers", "Integrated Telecommunication Services",
        "Wireless Telecommunication Services",
    ],
    "Media & Entertainment": [
        "Advertising", "Broadcasting", "Cable & Satellite", "Publishing",
        "Movies & Entertainment", "Interactive Home Entertainment", "Interactive Media & Services",
    ],
    # Utilities
    "Utilities": [
        "Electric Utilities", "Gas Utilities", "Multi-Utilities", "Water Utilities",
        "Independent Power Producers & Energy Traders", "Renewable Electricity",
    ],
    # Real Estate
    "Equity Real Estate Investment Trusts (REITs)": [
        "Diversified REITs", "Industrial REITs", "Hotel & Resort REITs", "Office REITs",
        "Health Care REITs", "Residential REITs", "Retail REITs", "Specialized REITs",
        "Telecom Tower REITs", "Timber REITs", "Data Center REITs",
        "Multi-Family Residential REITs", "Single-Family Residential REITs",
        "Self-Storage REITs", "Other Specialized REITs",
    ],
    "Real Estate Management & Development": [
        "Diversified Real Estate Activities", "Real Estate Operating Companies",
        "Real Estate Development", "Real Estate Services",
    ],
}

# Inverse lookup: normalized sub-industry name -> industry group.
_LOOKUP: dict[str, str] = {}
for _group, _subs in INDUSTRY_GROUP_MEMBERS.items():
    for _sub in _subs:
        _LOOKUP[_sub.lower()] = _group


def industry_group_for(sub_industry: str | None) -> str | None:
    """Return the GICS Industry Group for a Sub-Industry, or None if unmapped."""
    if not sub_industry or not isinstance(sub_industry, str):
        return None
    return _LOOKUP.get(sub_industry.strip().lower())
