import os 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from plots import PALETTE

canard = "#007480"
grosseile = "#b51f1f"
rouge = "#FF0000"
leman = "#00A79F"
acier = "#4F8FCC"
newcmp = ListedColormap([grosseile, acier])
classes_cmap = LinearSegmentedColormap.from_list("classes_cmap", PALETTE, N=len(PALETTE))
classes_cmap = ListedColormap(["#3274A1", "#E1812C", "#3A913A", "#C03D3E", "#886BA3", "#845B53", "#D584BD", "#7F7F7F", "#A9AA35"])
VMIN = 0
VMAX = 8

def plot_map_doy_stop(fig, ax, annot_fields, polygon, storepath, extension):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    annot_fields.plot(column="doy_stop", ax=ax, cmap="RdYlBu_r", vmin=1, 
                    vmax=365, legend=True, edgecolor="gray", legend_kwds={'label': 'day of stopping', "orientation": "vertical"})
    filename = os.path.join(storepath,"doy_stop."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved in ", filename)
    
def plot_map_correct(fig, ax, annot_fields, polygon, storepath, extension):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    annot_fields.plot(column="correct", ax=ax, cmap=newcmp)
    filename = os.path.join(storepath,"correct."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved at ", filename)
    
def plot_map_predictions_at_t_stop(fig, ax, annot_fields, polygon, classes_cmap, storepath, extension, vmin=VMIN, vmax=VMAX):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    annot_fields.plot(column="predictions_at_t_stop", ax=ax, cmap=classes_cmap, vmin=vmin, vmax=vmax)
    filename = os.path.join(storepath,"predictions."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved at ", filename)
    
def plot_map_targets(fig, ax, annot_fields, polygon, storepath, extension, vmin=VMIN, vmax=VMAX):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    annot_fields.plot(column="targets", ax=ax, cmap=classes_cmap, vmin=vmin, vmax=vmax)
    filename = os.path.join(storepath,"targets."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved at ", filename)
    
def plot_map_stop_date(fig, ax, annot_fields, date, polygon, storepath, stopped_cmap, extension="png"):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    annot_fields.plot(column="stopped_before_doy", ax=ax, cmap=stopped_cmap, edgecolor='gray')
    filename = os.path.join(storepath,f"stopped_at_date_{date}."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved at ", filename)
    
def plot_map_prediction_at_date(fig, ax, annot_fields, date, polygon, storepath, classes_cmap=classes_cmap, alpha=0.5, extension="png", vmin=VMIN, vmax=VMAX):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    if annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[annot_fields["stopped_before_doy"]].plot(column="prediction_at_doy", 
                                                                  ax=ax, cmap=classes_cmap, vmin=vmin, vmax=vmax)
    if ~annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[~annot_fields["stopped_before_doy"]].plot(column="prediction_at_doy", 
                                                                    ax=ax, cmap=classes_cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    else:
        annot_fields.plot(column="prediction_at_doy", ax=ax, cmap=classes_cmap, alpha=alpha, vmin=vmin, vmax=vmax)
         
    filename = os.path.join(storepath,f"prediction_at_date_{date}."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved at ", filename)
    
def plot_map_target_at_date(fig, ax, annot_fields, date, polygon, storepath, newcmp=newcmp, alpha=0.5, extension="png"):
    xmin, ymin, xmax, ymax = polygon.geometry.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    
    if annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[annot_fields["stopped_before_doy"]].plot(column="correct_at_doy", 
                                                                  ax=ax, cmap=newcmp, vmin=0, vmax=1)
    if ~annot_fields["stopped_before_doy"].sum() > 0:
        annot_fields.loc[~annot_fields["stopped_before_doy"]].plot(column="correct_at_doy", 
                                                                ax=ax, cmap=newcmp, alpha=alpha, vmin=0, vmax=1)
    else:
        annot_fields.plot(column="correct_at_doy", ax=ax, cmap=newcmp, alpha=alpha, vmin=0, vmax=1)
        
    filename = os.path.join(storepath,f"correct_at_date_{date}."+extension)
    fig.savefig(filename, bbox_inches="tight",transparent=True)
    print("figure saved at ", filename)

