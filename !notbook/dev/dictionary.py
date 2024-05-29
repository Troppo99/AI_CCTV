# Define the lists of people and activities
class_names_people = ["Alice", "Bob"]
class_names_activities = ["running", "walking"]


def manual(class_names_people, class_names_activities):
    # Initialize an empty dictionary
    time_accumulation = {}

    # Loop through each person in the list of people
    for person in class_names_people:
        # Create a new dictionary for each person
        activity_dict = {}

        # Loop through each activity in the list of activities
        for activity in class_names_activities:
            # Set the initial time spent on each activity to 0
            activity_dict[activity] = 0

        # Assign the dictionary of activities to the person
        time_accumulation[person] = activity_dict
    return time_accumulation


def simple(class_names_people, class_names_activities):
    time_accumulation = {
        person: {activity: 0 for activity in class_names_activities}
        for person in class_names_people
    }
    return time_accumulation


# Now time_accumulation is populated with the same structure as the comprehension
output = simple(class_names_people, class_names_activities)
print(output)
