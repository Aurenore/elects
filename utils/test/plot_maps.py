import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from PIL import Image
from plots import PALETTE

canard = "#007480"
grosseille = "#b51f1f"
rouge = "#FF0000"
leman = "#00A79F"
acier = "#4F8FCC"
newcmp = ListedColormap([grosseille, acier])
classes_cmap = LinearSegmentedColormap.from_list(
    "classes_cmap", PALETTE, N=len(PALETTE)
)
classes_cmap = ListedColormap(
    [
        "#3274A1",
        "#E1812C",
        "#3A913A",
        "#C03D3E",
        "#886BA3",
        "#845B53",
        "#D584BD",
        "#7F7F7F",
        "#A9AA35",
    ]
)
VMIN = 0
VMAX = 8


def save_patch_legend(storepath, classes_cmap, labels, file_sufix="", extension="png"):
    # Number of labels should match the number of discrete colors needed
    num_labels = len(labels)
    cmap = plt.get_cmap(
        classes_cmap, num_labels
    )  # Get discrete colors from the colormap

    # Create a figure and axes for displaying the legend
    fig, ax = plt.subplots(figsize=(2, 1 + 0.3 * num_labels))  # Adjust size as needed
    ax.axis("off")  # Turn off axis

    # Create a list of patches
    patches = []
    for i, label in enumerate(labels):
        color = cmap(i / num_labels)  # Get color for each label
        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)

    # Add the legend to the plot
    legend = ax.legend(handles=patches, loc="center", frameon=False, fontsize=12)

    # Save the legend as a separate file
    filename = os.path.join(storepath, f"legend{file_sufix}.{extension}")
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    print(f"Legend saved at {filename}")


def plot_map_doy_stop(fig, ax, annot_fields, polygon, storepath, extension):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    annot_fields.plot(
        column="doy_stop",
        ax=ax,
        cmap="RdYlBu_r",
        vmin=1,
        vmax=365,
        legend=True,
        edgecolor="gray",
        legend_kwds={"label": "day of stopping", "orientation": "vertical"},
    )
    filename = os.path.join(storepath, "doy_stop." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    print("figure saved in ", filename)


def plot_map_correct(fig, ax, annot_fields, polygon, storepath, extension):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    annot_fields.plot(column="correct", ax=ax, cmap=newcmp)
    filename = os.path.join(storepath, "correct." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    save_patch_legend(
        storepath,
        newcmp,
        ["incorrect", "correct"],
        file_sufix="_correct",
        extension=extension,
    )
    print("figure saved at ", filename)


def plot_map_predictions_at_t_stop(
    fig,
    ax,
    annot_fields,
    polygon,
    classes_cmap,
    storepath,
    extension,
    class_names,
    vmin=VMIN,
    vmax=VMAX,
):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    annot_fields.plot(
        column="predictions_at_t_stop", ax=ax, cmap=classes_cmap, vmin=vmin, vmax=vmax
    )
    filename = os.path.join(storepath, "predictions." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    save_patch_legend(
        storepath,
        classes_cmap,
        class_names,
        file_sufix="_predictions",
        extension=extension,
    )
    print("figure saved at ", filename)


def plot_map_targets(
    fig, ax, annot_fields, polygon, storepath, extension, vmin=VMIN, vmax=VMAX
):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    annot_fields.plot(column="targets", ax=ax, cmap=classes_cmap, vmin=vmin, vmax=vmax)
    filename = os.path.join(storepath, "targets." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    print("figure saved at ", filename)


def plot_map_stop_date(
    fig, ax, annot_fields, date, polygon, storepath, stopped_cmap, extension="png"
):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    annot_fields.plot(
        column="stopped_before_doy", ax=ax, cmap=stopped_cmap, edgecolor="gray"
    )
    filename = os.path.join(storepath, f"stopped_at_date_{date}." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    save_patch_legend(
        storepath,
        stopped_cmap,
        ["not stopped", "stopped"],
        file_sufix="_stopped",
        extension=extension,
    )
    print("figure saved at ", filename)


def plot_map_prediction_at_date(
    fig,
    ax,
    annot_fields,
    date,
    polygon,
    storepath,
    classes_cmap=classes_cmap,
    alpha=0.5,
    extension="png",
    vmin=VMIN,
    vmax=VMAX,
):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    if annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[annot_fields["stopped_before_doy"]].plot(
            column="prediction_at_doy", ax=ax, cmap=classes_cmap, vmin=vmin, vmax=vmax
        )
    if ~annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[~annot_fields["stopped_before_doy"]].plot(
            column="prediction_at_doy",
            ax=ax,
            cmap=classes_cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        annot_fields.plot(
            column="prediction_at_doy",
            ax=ax,
            cmap=classes_cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )

    filename = os.path.join(storepath, f"prediction_at_date_{date}." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    print("figure saved at ", filename)


def plot_map_target_at_date(
    fig,
    ax,
    annot_fields,
    date,
    polygon,
    storepath,
    newcmp=newcmp,
    alpha=0.5,
    extension="png",
):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    if annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[annot_fields["stopped_before_doy"]].plot(
            column="correct_at_doy", ax=ax, cmap=newcmp, vmin=0, vmax=1
        )
    if ~annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[~annot_fields["stopped_before_doy"]].plot(
            column="correct_at_doy", ax=ax, cmap=newcmp, alpha=alpha, vmin=0, vmax=1
        )
    else:
        annot_fields.plot(
            column="correct_at_doy", ax=ax, cmap=newcmp, alpha=alpha, vmin=0, vmax=1
        )

    filename = os.path.join(storepath, f"correct_at_date_{date}." + extension)
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    print("figure saved at ", filename)


def plot_map_table(
    storepath, dates_of_interest, sat_pics_path, extension="png", fontsize=18
):
    sat_pics_filenames = os.listdir(sat_pics_path)

    print("directory: ", storepath)
    # Define the image file names for each category
    dates = dates_of_interest
    categories = {
        "Satellite Images": sat_pics_filenames + ["targets.png"],
        "Prediction": ["prediction_at_date_{}.png".format(date) for date in dates]
        + ["predictions.png"],
        "Correct/Incorrect": ["correct_at_date_{}.png".format(date) for date in dates]
        + ["correct.png"],
        "Stopped": ["stopped_at_date_{}.png".format(date) for date in dates]
        + ["doy_stop.png"],
    }
    last_col_labels = [
        "Ground Truth",
        "Prediction at stop",
        "In/correct at stop",
        "Day of Stopping",
    ]

    # Initialize the plot
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(
        4, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.5]
    )  # Reduce the last column width

    axs = [[fig.add_subplot(gs[row, col]) for col in range(5)] for row in range(4)]
    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    # Function to load an image
    def load_image(filename, storepath=storepath):
        if filename is None:
            return None  # Return None if there is no file to load (e.g., empty slots in the grid)
        path = os.path.join(storepath, filename)
        return Image.open(path)

    # Place images in the grid
    for row, (label, files) in enumerate(categories.items()):
        fig.text(
            0.005,
            0.88 - (row / 4),
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            transform=plt.gcf().transFigure,
            rotation=90,
        )
        for col, file in enumerate(files):
            if row == 0 and col < len(dates):
                img = load_image(file, sat_pics_path)
                if img is not None:
                    axs[row][col].imshow(img)
                    axs[row][col].set_title("{}".format(dates[col]), fontsize=fontsize)
            elif file is not None:  # Check if there is a file to display
                img = load_image(file, storepath)
                if img is not None:
                    axs[row][col].imshow(img)
                    if col == 3:
                        axs[row][col].set_title(
                            "{}".format(last_col_labels[row]), fontsize=fontsize
                        )

    # place legends in the last column
    legend_files = [
        "legend_predictions.png",
        "legend_correct.png",
        "legend_stopped.png",
    ]
    for row, file in enumerate(legend_files):
        img = load_image(file, storepath)
        if img is not None:
            axs[row + 1][-1].imshow(img)

    # Adjust layout
    fig.tight_layout()
    fig.savefig(os.path.join(storepath, "all_maps." + extension))
    plt.show()
