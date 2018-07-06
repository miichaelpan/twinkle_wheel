import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw
import cv2
import math
import random
from types import SimpleNamespace
import moviepy.editor as mpy

#basic image properties
image_viewport_w = 53
image_viewport_h = 53
image_format = 'RGB'
fps = 2
rotation_per_frame = 2
file_name = "draw_test_"+str(rotation_per_frame)+".jpg"


#plot properties and initialization
fibers= []
coord = []
n_drops = 0
delay_in_ms = 150




# main function
def main():
    # configure properties per input image resolution

    #wheel 1
    # input_image = 'wheel(780x780).png'
    # input_image_w = 780
    # input_image_h = 780

    # wheel 2, need to adjust threshold and contouring
    input_image = 'wheel2(790x790).png'
    input_image_w = 790
    input_image_h = 790

    # wheel 3,
    # input_image = 'wheel3(720x720).png'
    # input_image_w = 720
    # input_image_h = 720

    #wheel 4, 
    # input_image = 'wheel4(720x720).png'
    # input_image_w = 720
    # input_image_h = 720


    # Get viewport and normalize
    viewport_image = get_viewport_image(input_image_w, input_image_h, image_viewport_w, image_viewport_h)
    # viewport_image.show()


    # Get contour image from twinkle wheel picture
    contour_image = get_contour_image(input_image, input_image_w, input_image_h)
    contour_image.resize((int(input_image_w/2), int(input_image_h/2)), Image.ANTIALIAS).show()


    # Iterate through the frames given rotation_per_frame(degrees)
    frames, frames_rand, intensity_ratio = get_frames(viewport_image, contour_image, frame_format="PIL")
    # print(intensity_ratio)
    # print(frames)
    # frames2 = Image.fromarray(frames[0], 'L')
    # frames2.show()

    # listy = get_gif_list(file_name, frames)

    get_intensity_ratio_plot(intensity_ratio)
    # print(intensity_ratio)
    generate_plot(viewport_image, frames, frames_rand)


#get the contour of the input image, use blur for better results
def get_contour_image(input_image, input_image_w, input_image_h):
    img = cv2.imread(input_image, 0)
    img = cv2.medianBlur(img, 5)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_nparray = np.zeros((input_image_w, input_image_h, 3), np.uint8)
    for i in range(len(contours)-2):
        cv2.drawContours(blank_nparray, contours, len(contours)-i-1, (255,255,255), -1)
    contour_image = Image.fromarray(blank_nparray, image_format)
    return contour_image




# get the circlular image mask based on resolution
def get_mask_image(image_w, image_h):
    image1 = Image.new(image_format, (image_w, image_h))
    draw = ImageDraw.Draw(image1)
    ellipse_tuple = (0, 0, image_w, image_h)
    color1_fill = (255, 255, 255)
    color1_outline = (255, 255, 255)
    draw.ellipse(ellipse_tuple, color1_fill, color1_outline)
    # image1.show()
    return image1
    

def get_normalized_image(image, image_viewport_h, image_viewport_w):
    normalized_image = image.resize((image_viewport_h, image_viewport_w), Image.ANTIALIAS)
    # normalized_image.show()
    return normalized_image

# Resize the mask image to be the size of the viewport
def get_viewport_image(input_image_h, input_image_w, image_viewport_h, image_viewport_w):
    image_mask = get_mask_image(input_image_w, input_image_h)
    normalized_image_mask = get_normalized_image(image_mask, image_viewport_h, image_viewport_w)
    # normalized_image_mask.show()
    return normalized_image_mask



# Get list of total lit(valid) pixels in the image
def get_list_of_total_shown_pixels(base_image):
    total_pixels_list = []
    np_image = np.array(base_image.convert('L'))
    count_of_white_pixels = 0
    for x_point in range(image_viewport_h):
        for y_point in range(image_viewport_w):
            if np_image[x_point, y_point] > 254/2:
                count_of_white_pixels += 1
                index = (x_point * image_viewport_w) + y_point
                # index = get_encoded_coord(x_point, y_point, image_viewport_w, image_viewport_h)
                total_pixels_list.append(index)
                # print(total_pixels_list)
    return total_pixels_list


# def get_encoded_coord(x_point, y_point, image_w, image_h):
#     encoded_value = (x_point * image_h) + y_point
#     return encoded_value

# def get_encoded_coord2(coded_index, image_w, image_h):
#     x_point = int(coded_index % image_h)
#     y_point = int((coded_index - y_point) / image_w)
#     return x_point, y_point
    





# Get array of all valid pixels in random positions - fiber bundle
def get_random_series(base_image):
    total_pixels = get_list_of_total_shown_pixels(base_image)
    random_map = random.sample(total_pixels, len(total_pixels))
    new_random_map = get_image_map(base_image, random_map)
    return new_random_map


# Get randomized image map from base image
def get_image_map(base_image, random_map):
    new_f = np.zeros((image_viewport_h, image_viewport_w), int)
    np_image = np.array(base_image.convert('L'))
    counter = 0
    for x_point in range(image_viewport_w):
        for y_point in range(image_viewport_h):
            if np_image[x_point, y_point] > 256/2:
                new_f[x_point, y_point] = random_map[counter]
                counter += 1
            else:
                new_f[x_point, y_point] = 0
    return new_f



def get_frames(norm_image, contour_image, frame_format="PIL"):
    frames = []
    frames_rand = []
    intensity_ratio = []
    # Get the random mapping
    random_map_np = get_random_series(norm_image)
    # Size up viewport image into array
    viewport_np = np.array(norm_image.convert('L'))
    viewport_h = len(viewport_np)
    viewport_w = len(viewport_np[0])
    # Size up contour image into array
    contour_np = np.array(contour_image.convert('L'))
    contour_h = len(contour_np)
    contour_w = len(contour_np[0])
    for i in range(int(360/rotation_per_frame)):
        degrees = int(i * rotation_per_frame)
        normalized_disk_image, nothing = get_disk_image(contour_image, degrees, contour_w, contour_h, image_viewport_w, image_viewport_h)
        binary_normalized_image = np.array(normalized_disk_image.convert('L'))
        # Create np.zero arrays for new frames
        new_f = np.zeros((image_viewport_w, image_viewport_h), np.uint8)
        new_f_rand = np.zeros((image_viewport_w, image_viewport_h), np.uint8)
        # Initialize pixel count
        count_of_white_pixels = 0
        count_of_intensity = 0
        for x_point in range(image_viewport_w):
            for y_point in range(image_viewport_h):
                if viewport_np[x_point, y_point] > 256/2:
                    count_of_white_pixels += 1
                    if binary_normalized_image[x_point, y_point] > 256/2:
                        count_of_intensity += 1
                        new_f[x_point, y_point] = 255
                        coded_index = random_map_np[x_point, y_point]
                        random_x_point = int(coded_index % image_viewport_h)
                        random_y_point = int((coded_index - y_point) / image_viewport_w)
                        new_f_rand[random_x_point, random_y_point] = 255
                    else:
                        coded_index = random_map_np[x_point, y_point]
                        new_f[x_point,y_point] = 0
                else:
                    new_f[x_point, y_point] = 0
                    new_f_rand[x_point, y_point] = 0
        # Will need function to convert from array to L to RGB and flip
        # for both frames and frames_rand
        frames.append(new_f)
        frames_rand.append(new_f_rand)
        intensity_count = count_of_intensity/count_of_white_pixels * 100
        intensity_ratio.append(float('{0:.2f}'.format(intensity_count)))
        # print(intensity_ratio)
        # print(count_of_white_pixels)
    return frames, frames_rand, intensity_ratio



# Get normalized contour image and use this in the for loop inside get_frames
def get_disk_image(contour_image, degrees, image_w, image_h, image_viewport_w, image_viewport_h):
    disk_image = contour_image.rotate(-1 * degrees)
    # disk_image.show()
    viewport_tuple = (10,image_h*(1.7/5),image_w*(2/5)+10,image_h*(3.7/5))
    disk_image = disk_image.crop(viewport_tuple)
    norm_disk_image = disk_image.resize((image_viewport_w, image_viewport_h), Image.ANTIALIAS)
    # disk_image.show()
    return norm_disk_image, disk_image



def generate_plot(base_image, frames, frames_rand):
    plt.style.use('dark_background')
    total_pixels_list = get_list_of_total_shown_pixels(base_image)
    n_drops = len(total_pixels_list)
    # Re-calculate because not a perfect square for plotting
    n_drops = int(math.sqrt(n_drops)) * int(math.sqrt(n_drops))
    # Initialize the window size (inches)
    plot_size_w = 5.9
    plot_size_h = 5.9
    intensity = 10
    pixel_distance = 0.02
    fig = plt.figure(figsize=(plot_size_w, plot_size_h))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('equal')
    ax.set_xlim(0,1), ax.set_xticks([])
    ax.set_ylim(0,1), ax.set_yticks([])

    coord = []
    for x in range(int(math.sqrt(n_drops))):
        for y in range(int(math.sqrt(n_drops))):
            coord.append([pixel_distance * (x + 1), pixel_distance * (y+1)])

    # Create fibers
    fibers = np.zeros(n_drops, dtype = [('position', float, 2),
                                    ('size', float, 1),
                                    ('growth', float, 1),
                                    ('color', float, 4)])

    # Construct scatter plot
    scat = ax.scatter(fibers['position'][:,0], fibers['position'][:,1],
                    s=fibers['size'], lw=0.5, edgecolors=fibers['color'],
                    facecolors='white')
    
    fibers['color'] = (0, 0, 0, 0)

    # Scatter plot points
    def update(frame_number):
        frame_index = frame_number % len(frames_rand)
        current_frame = frames_rand[frame_index]
        for drop in range(n_drops):
            coded_index = total_pixels_list[drop]
            frame_h = len(current_frame)
            frame_w = len(current_frame[0])
            random_y_point = int(coded_index % frame_h)
            random_x_point = int((coded_index - random_y_point) / frame_w)
            dot_size = fibers['size'][drop]  - int(intensity/2)
            if current_frame[random_x_point, random_y_point] != 0:
                dot_size = 1 * intensity
            fibers['color'][drop] = (1, 1, 1, 1)
            fibers['growth'][drop] = intensity/3
            fibers['size'][drop] = dot_size
            fibers['position'][drop] = coord[drop]
        
        # Update the scatter collection with the new colors, size, and positions
        scat.set_edgecolors(fibers['color'])
        scat.set_sizes(fibers['size'])
        scat.set_offsets(fibers['position'])

    animation = FuncAnimation(fig, update, interval = delay_in_ms)
    plt.show()


def get_intensity_ratio_plot(intensity_ratio):
    x = []
    for i in range(int(360/rotation_per_frame)):
        degrees = int(i * rotation_per_frame)
        x.append(degrees)
    
    y = intensity_ratio
    plt.figure(1)
    # print(len(x))
    # print(len(y))
    plt.axes()
    plt.ylim([0, 100])
    plt.title('Twinkle Wheel Intensity Profile @ Fiberport')
    plt.xticks(np.arange(0,361, 180))
    plt.xlabel('rotation (deg)')
    plt.ylabel('intensity ratio (%)')
    plt.plot(x, y, color = 'b')
    # plt.show()


def get_gif_list(file_name, frames):
    gif_list = []
    
    for frame in frames:
        for i in range(int(360/rotation_per_frame)):
            pixels = Image.fromarray(frame, 'L')
            pixels = np.array(pixels.convert('RGB') )
            pixels= pixels[:, :, ::-1].copy()
            gif_list.append(pixels)
            degrees = int(i * rotation_per_frame) 
            file_name = "draw_test_"+str(degrees)+".jpg"
            cv2.imwrite(file_name, gif_list)
        # print(type(pixels))
    # print(gif_list)
    return gif_list







        # pixels[5].show
    # for i in gif_list:
        # cv2.imwrite(file_name, gif_list)

        # for pixel in pixels:
        #     open_cv_image = np.array(pixel.convert('RGB') ) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # frames.append(open_cv_image)
        # gif_list = frames.append(pixels)
        # clips = [mpy.ImageClip(m).set_duration(1.0/fps) for m in frames]
        # concat_clip = mpy.concatenate_videoclips(clips, method="compose")
        # concat_clip.write_videofile("{}.mp4".format(file_name), fps=fps)
    gif_list[10].show()



main()

