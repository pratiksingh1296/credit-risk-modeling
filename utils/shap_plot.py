import matplotlib.pyplot as plt


def plot_shap_waterfall(feature_names, shap_values, base_value, max_display=10):
    
    # Pair feature names with SHAP values
    features = list(zip(feature_names, shap_values))

    # Sort by absolute contribution
    features.sort(key=lambda x: abs(x[1]), reverse=True)

    # Keep top features
    top_features = features[:max_display]

    # Combine remaining features
    remaining = features[max_display:]

    if remaining:
        other_value = sum(value for _, value in remaining)
        top_features.append(
            (f"Other {len(remaining)} features (combined)", other_value)
        )

    # Largest contribution at top
    top_features = top_features[::-1]

    names = [name for name, _ in top_features]
    values = [value for _, value in top_features]

    # Calculate cumulative positions
    cumulative = base_value
    bars = []

    for name, value in zip(names, values):
        start = cumulative
        end = cumulative + value

        bars.append((name, start, end, value))
        cumulative = end

    # Final model output
    final_value = cumulative

    # Colors
    POS_COLOR = "#E84A5F"
    NEG_COLOR = "#3B82C4"
    OTHER_COLOR = "#9CA3AF"

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Draw bars
    for name, start, end, value in bars:

        left = min(start, end)
        width = abs(value)

        if "other features" in name.lower():
            color = OTHER_COLOR
        else:
            color = POS_COLOR if value >= 0 else NEG_COLOR

        ax.barh(
            name,
            width,
            left=left,
            height=0.58,
            color=color
        )

        # Value labels
        if abs(value) >= 0.01:
            label_x = start + value / 2

            ax.text(
                label_x,
                name,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="semibold"
            )

    # Zero reference line
    ax.axvline(
        0,
        color="#9CA3AF",
        linewidth=1,
        alpha=0.8
    )

    # Base value reference line
    ax.axvline(
        base_value,
        linestyle="--",
        linewidth=1,
        color="#6B7280",
        alpha=0.8
    )

    # -------------------------
    # Clean styling
    # -------------------------

    ax.grid(False)

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Y-axis labels
    ax.tick_params(
        axis="y",
        labelsize=9,
        colors="#374151",
        length=0
    )

    # X-axis labels
    ax.tick_params(
        axis="x",
        labelsize=9,
        colors="#6B7280",
        length=3
    )

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Subtle baseline
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.spines["bottom"].set_linewidth(0.8)

    # Layout
    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        top=0.95,
        bottom=0.15
    )

    return fig
