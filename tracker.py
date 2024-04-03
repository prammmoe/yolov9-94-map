import math

class CustomTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.custom_center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.custom_id_count = 0

    def custom_update(self, custom_objects_rect):
        # Objects boxes and ids
        custom_objects_bbs_ids = []

        # Get center point of new object
        for custom_rect in custom_objects_rect:
            x, y, w, h = custom_rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for custom_id, pt in self.custom_center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.custom_center_points[custom_id] = (cx, cy)
                    custom_objects_bbs_ids.append([x, y, w, h, custom_id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.custom_center_points[self.custom_id_count] = (cx, cy)
                custom_objects_bbs_ids.append([x, y, w, h, self.custom_id_count])
                self.custom_id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_custom_center_points = {}
        for custom_obj_bb_id in custom_objects_bbs_ids:
            _, _, _, _, custom_object_id = custom_obj_bb_id
            center = self.custom_center_points[custom_object_id]
            new_custom_center_points[custom_object_id] = center

        # Update dictionary with IDs not used removed
        self.custom_center_points = new_custom_center_points.copy()
        return custom_objects_bbs_ids