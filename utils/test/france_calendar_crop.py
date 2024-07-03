import datetime
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Crop': ["barley", "wheat", "rapeseed", "corn", "sunflower"],
    'Plant Start': [9, 9.5, 8.5, 4, 4],
    'Plant End': [11, 12, 10.5, 6, 6],
    'Mid-Season Start': [11, 1, 10.5, 6, 6],
    'Mid-Season End': [13, 6.5, 13, 9, 8],
    'Second Mid-Season Start': [1, 12, 1, 0, 0],
    'Second Mid-Season End': [6, 13, 6, 0, 0],
    'Harvest Start': [6, 6.5, 6, 9, 8],
    'Harvest End': [8, 9, 8, 12, 10]
}
colors_calendar = ['#dde4d1', '#c7c7c7', '#f5e1aa']

def month_to_day_of_year(month, year):
    if not 1 <= month <= 13:
        raise ValueError("Month must be between 1 and 13")
    if year < 1:
        raise ValueError("Year must be a positive integer")
    
    if month == 13:
        month = 12
        day = 31
    elif month % 1 != 0:
        day = int(month % 1 * 31)
    else:
        day = 1
    month = int(month)
    
    # Calculate the first day of the given month
    given_day = datetime.date(year, month, day)
    
    # Return the day of the year
    return given_day.timetuple().tm_yday

def add_crop_calendar(ax, labels_names):
    df = pd.DataFrame(data)
    # transform the number of month to day of year
    columns_date = ['Plant Start', 'Plant End', 'Mid-Season Start', 'Mid-Season End', 'Second Mid-Season Start', 'Second Mid-Season End', 'Harvest Start', 'Harvest End']
    df[columns_date] = df[columns_date].map(lambda x: month_to_day_of_year(x, 2017) if x != 0 else 0)
    shift = -0.3
    width = abs(shift) * 2
    yticks = ax.get_yticks()
    for i, y_label in enumerate(labels_names):
        if y_label in df['Crop'].values:
            # add the rectangle at yticks[i]+shift
            # get the row of the crop
            row = df[df['Crop'] == y_label].iloc[0]  # Ensure only one row is indexed
            # Plant phase
            ax.add_patch(plt.Rectangle((row['Plant Start'], yticks[i]+shift), row['Plant End'] - row['Plant Start'], width, color=colors_calendar[0], label='Plant' if i == 0 else ""))
            # Mid-Season phase
            ax.add_patch(plt.Rectangle((row['Mid-Season Start'], yticks[i]+shift), row['Mid-Season End'] - row['Mid-Season Start'], width, color=colors_calendar[1], label='Mid-Season' if i == 0 else ""))
            ax.add_patch(plt.Rectangle((row['Second Mid-Season Start'], yticks[i]+shift), row['Second Mid-Season End'] - row['Second Mid-Season Start'], width, color=colors_calendar[1]))
            # Harvest phase
            ax.add_patch(plt.Rectangle((row['Harvest Start'], yticks[i]+shift), row['Harvest End'] - row['Harvest Start'], width, color=colors_calendar[2], label='Harvest' if i == 0 else ""))
    # place the legend at the bottom right
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=14)
    return ax