from PIL import Image, ImageDraw
import cv2
import numpy as np
import moviepy.editor as mpy
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# properties
image_h = 780
image_w = 780
image_viewport_h = 53
image_viewport_w = 53
image_fiber_h = 47
image_fiber_w = 47
image_format = 'RGB'
# ellipse1_tuple = (0,0,image_h,image_w)
ellipse1_tuple = (0,0,image_viewport_h,image_viewport_w)
fiber_tuple = (0,0,image_fiber_h-1,image_fiber_w-1)
fiber_port_tuple = (10,image_h*(1.7/5),image_w*(2/5)+10,image_h*(3.7/5))
color1_fill = (255,255,255)
color1_outline = (255,255,255)
file_name_of_twinkle_wheel = 'wheel(780x780).png'
file_name_of_output_mp4 = 'TW_spin_gif2'
fps = 2
delta_rotation = 5




# get contour image from input file
def get_contour_image(file_name):
    img = cv2.imread(file_name,0)
    # blur the image for better edge results
    img = cv2.medianBlur(img,5)
    # set the threshold for the contours (127,255) is good for B&W
    ret, thresh = cv2.threshold(img,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # define blank array similar to size of input image resolution
    blank_nparray = np.zeros((image_h,image_w,3), np.uint8)
    # draw the contours onto the nparray
    #cv2.drawContours(blank_nparray, contours, -1, (0,255,0), -1) # this only draws contour outlines
    #use for loop to fill in dark spaces
    for i in range(len(contours)-2):
        cv2.drawContours(blank_nparray, contours, len(contours)-i-1, (0,255,0), -1)
    # convert the nparray to an image using imageio
    return Image.fromarray(blank_nparray, image_format)




#get rotational images
def get_frames(norm_image, contour_image):
    # initialize list
    frames = []
    frames_rand = []
    intensity_ratio = []
    random_map_np = get_random_series(norm_image)
    norm_image_mask = np.array(norm_image.convert('L'))
    
    # print(random_map_np)
    # pixels = Image.fromarray(random_map_np, 'L')
    # pixels.show()
    for i in range(int(360/delta_rotation)):
        degrees = int(i * delta_rotation)
        image_frame = contour_image.rotate(-1*degrees)
        image_frame = image_frame.crop(fiber_port_tuple)
        image_frame = image_frame.resize((image_h, image_w), Image.ANTIALIAS)
        norm_image_frame = image_frame.resize((image_viewport_h, image_viewport_w), Image.ANTIALIAS)
        # result = Image.composite(norm_image_frame, norm_image_mask, norm_image_mask)
        binary_norm_image = np.array(norm_image_frame.convert('L'))

        new_f = np.zeros((image_viewport_h, image_viewport_w), np.uint8)
        new_f_rand = np.zeros((image_viewport_h, image_viewport_w), np.uint8)
        count_of_white_pixels = 0
        count_of_intensity = 0
        for x_point in range(image_viewport_w):
            for y_point in range(image_viewport_h):
                # print(binary_norm_image[x_point,y_point])
                if norm_image_mask[x_point,y_point] > 256/2:
                    count_of_white_pixels += 1
                    if binary_norm_image[x_point,y_point] > 256/2:
                        count_of_intensity += 1
                        new_f[x_point,y_point] = 255

                        coded_index = random_map_np[x_point,y_point]
                        # print(coded_index)
                        random_y_point = int(coded_index % image_viewport_h)
                        random_x_point = int((coded_index - random_y_point) / image_viewport_w)
                        # print(new_f_rand[random_x_point,random_y_point])
                        new_f_rand[random_x_point,random_y_point] = 255
                    else:
                        coded_index = random_map_np[x_point,y_point]
                        # print(coded_index)
                        new_f[x_point,y_point] = 0
                else:
                    new_f[x_point,y_point] = 0
                    new_f_rand[x_point,y_point] = 0


        # pixels = Image.fromarray(new_f, 'L')
        # open_cv_image = np.array(pixels.convert('RGB') ) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # frames.append(open_cv_image)
        frames.append(new_f)
        # Write frame of view port to file
        # file_name = "draw_test_"+str(degrees)+".jpg"
        # cv2.imwrite(file_name, open_cv_image)
        # cv2.imwrite(file_name, new_f)
        
        # pixels = Image.fromarray(new_f_rand, 'L')
        # open_cv_image = np.array(pixels.convert('RGB') ) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # frames_rand.append(open_cv_image)
        frames_rand.append(new_f_rand)
        # if i == 0:
        #     pixels = Image.fromarray(new_f_rand, 'L')
        #     pixels.show()
        #intensity_ratio = (100*count_of_intensity/count_of_white_pixels)
        intensity_ratio.append(float("{0:.2f}".format(100*count_of_intensity/count_of_white_pixels)))
        #print("INTENSITY: {0:.2f}%".format(intensity_ratio[-1]))
        # n_white_pix = np.sum(test_frame == 255)
        # # Convert to RGB to BGR
        # open_cv_image = np.array(result.convert('RGB')) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # frames.append(open_cv_image)
        
        # Write random frame to file if desired
        #file_name = "rand_draw_test_"+str(degrees)+".jpg"
        #cv2.imwrite(file_name, open_cv_image)
        #print(n_white_pix)
    #print(intensity_ratio)
    # graph with gnuplot.py


    # # create image of the fiber bundle using pixels
    # image3 = Image.new(image_format,(image_fiber_h, image_fiber_w))
    # #image3 = Image.new(image_format, (image_h, image_w))
    # draw = ImageDraw.Draw(image3)
    # draw.ellipse(fiber_tuple, fill=color1_fill, outline=color1_outline)
    # image3 = image3.convert('L')
    # #image3.show()


    # # Map pixels to a random list of numbers
    # random_map = random.sample(range(-1, (image_fiber_h * image_fiber_w)), image_fiber_h * image_fiber_w)
    # #random_map = random.sample(range(-1, (image_w * image_h)), image_w*image_h-10)
    # old_f = np.array(image3.convert('L'))
    # new_f = np.zeros((image_fiber_h, image_fiber_w), np.uint8)
    # for x_point in range(image_fiber_w):
    #     for y_point in range(image_fiber_h):
    #         random_map_key = (x_point * image_fiber_h) + y_point
    #         random_map_value = random_map[random_map_key]
    #         #random_map_value = (x_point * image_fiber_h) + y_point #don't use
    #         random_y_point = int(random_map_value % image_fiber_h)
    #         random_x_point = int((random_map_value - random_y_point) / image_fiber_w)
    #         new_f[x_point,y_point] = old_f[random_x_point,random_y_point]
    # pixels = Image.fromarray(new_f, 'L')
    # pixel_port = Image.composite(pixels, image3, image3)
    # #pixels.show()
    # pixel_port.show()



    # # Find number of white pixels of the open fiber port using blank image
    # img_test = cv2.imread('open_port.jpg', cv2.IMREAD_GRAYSCALE)
    # n_white_pix = np.sum(img_test == 255)
    # #print('Number of white pixels in open port:', n_white_pix)
    return frames, frames_rand, intensity_ratio





# Use function to get rotational image of wheel and fiber bundle
# But, how to create fiber bundle
#def get_fiber_port(image3, image2):
#    frames2 = []
#    for i in range(int(360/delta_rotation)):
#        degrees = int(i * delta_rotation)
#        image_frame = image2.rotate(-1*degrees)
#        blank_fiber_array = np.zeros((image_h,image_w,3), np.uint8)
#        image_mask = image3.convert('L')
#        image_frame = image2
#        image_frame = image_frame.resize((image_h, image_w), Image.ANTIALIAS)
#        result = Image.composite(image_frame, image_mask, image_mask)
#        # Convert to RGB to BGR
#        open_cv_image = np.array(result.convert('RGB') ) 
#        open_cv_image = open_cv_image[:, :, ::-1].copy()
#        frames.append(open_cv_image)
        # Write each frame to file if desired
        #file_name = "draw_test_"+str(degrees)+".jpg"
        #cv2.imwrite(file_name, open_cv_image)
        #print(n_white_pix)
#    return frames2




# Make gif file 
def make_gif(file_name, frames, fps=30):
    clips = [mpy.ImageClip(m).set_duration(1.0/fps) for m in frames]
    concat_clip = mpy.concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile("{}.mp4".format(file_name), fps=fps)








def get_mask():
    # get the first image (this is the light port) and normalize it to the viewport image
    # image1 = Image.new(image_format, (image_w, image_h))
    # draw = ImageDraw.Draw(image1)
    # draw.ellipse(ellipse1_tuple, fill=color1_fill, outline=color1_outline)
    # image1 = image1.resize((image_viewport_h, image_viewport_w), Image.ANTIALIAS)

    image1 = Image.new(image_format, (image_viewport_w, image_viewport_h))
    draw = ImageDraw.Draw(image1)
    draw.ellipse(ellipse1_tuple, fill=color1_fill, outline=color1_outline)
    return image1

# # Normalized image of the base image (input)
# def get_normalized_image(base_image):
#     image_frame = base_image.resize((image_viewport_h, image_viewport_w), Image.ANTIALIAS)
#     return image_frame





# Total number of pixels
def get_list_of_total_of_shown_pixels(base_image):
    total_pixels_list = []
    np_image = np.array(base_image.convert('L'))
    count_of_white_pixels = 0
    for x_point in range(image_viewport_w):
        for y_point in range(image_viewport_h):
            # print(np_image[x_point,y_point])
            if np_image[x_point,y_point] > 256/2:
                count_of_white_pixels += 1
                # index = (x_point * image_viewport_w) + y_point
                index = get_encoded_coord(x_point, y_point, image_viewport_w, image_viewport_h)
                total_pixels_list.append(index)
    return total_pixels_list




# Pixel locations
def get_image_map(base_image,random_map):
    new_f = np.zeros((image_viewport_h, image_viewport_w), int)
    np_image = np.array(base_image.convert('L'))
    counter = 0
    for x_point in range(image_viewport_w):
        for y_point in range(image_viewport_h):
            if np_image[x_point,y_point] > 256/2:
                new_f[x_point,y_point] = random_map[counter]
                counter += 1
            else:
                new_f[x_point,y_point] = 0
    return new_f





def get_random_series(base_image):
    # this will return array of all valid pixels with random positions
    total_pixels = get_list_of_total_of_shown_pixels(base_image)
    random_map = random.sample(total_pixels, len(total_pixels))
    new_map = get_image_map(base_image,random_map)
    return new_map




def get_masked_list(base_image):
    # assumes base_image is of length and with of viewport
    ignore_list = []
    np_image = np.array(base_image.convert('L'))
    new_f = np.zeros((image_fiber_h, image_fiber_w), np.uint8)
    count_of_white_pixels = 0
    for x_point in range(image_viewport_w):
        for y_point in range(image_viewport_h):
            # print(np_image[x_point,y_point])
            if np_image[x_point,y_point] > 256/2:
                count_of_white_pixels += 1
                ignore_list.append(1)
            else:
                ignore_list.append(0)
    # print("PIXELS TOTAL ",count_of_white_pixels)
    return ignore_list
            # random_map_key = (x_point * image_fiber_h) + y_point
            # random_map_value = random_map[random_map_key]
            # #random_map_value = (x_point * image_fiber_h) + y_point #don't use
            # random_y_point = int(random_map_value % image_fiber_h)
            # random_x_point = int((random_map_value - random_y_point) / image_fiber_w)
            # new_f[x_point,y_point] = np_image[random_x_point,random_y_point]




def main():
    # makes viewport image (this is your image mask)
    image_mask = get_mask() 
    # norm_image_mask = get_normalized_image(image_mask)
    # ignore_list = get_masked_list(norm_image_mask)
    ignore_list = get_masked_list(image_mask)

    contour_image = get_contour_image(file_name_of_twinkle_wheel)
    #frames, frames_rand = get_frames(norm_image_mask, contour_image)
    frames, frames_rand, intensity_ratio, random_map_np = get_frames(image_mask, contour_image)
    #make_gif(file_name_of_output_mp4, frames, fps)
    #make_gif(file_name_of_output_mp4+"_rand", frames_rand, fps)

    ## Find numer of white pixels from saved image
    ## Want to make a loop so that it does this for all the images
    ## currently only works with one input image
    #img_test2 = cv2.imread('draw_test_10.jpg', cv2.IMREAD_GRAYSCALE)
    #ret, thresh2 = cv2.threshold(img_test2,127,255,0)
    #im2, contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #n_white_pix2 = np.sum(thresh2 == 255)
    #print('Number of white pixels in test frame:', n_white_pix2)
    return frames, frames_rand, intensity_ratio, random_map_np




#def intensity_ratio():
#    input_image = []
#    thresh_list = []
#    n_white_pix2 = []
#    output = []
#    for i in range(int(360/delta_rotation)):
#            degrees = int(i * delta_rotation)
#            read_file_name = "draw_test_"+str(degrees)+".jpg"
#            input_image = cv2.imread(read_file_name, cv2.IMREAD_GRAYSCALE)
#            ret, thresh_list = cv2.threshold(input_image,127,255,0)
#            n_white_pix2 = np.sum(thresh_list == 255)
#            n_white_pix2 = np.int(n_white_pix2)
#            output.append(n_white_pix2)
#            return output


# def get_intensity():
#     intensity_ratio_list = []
#     n_white_pix2 = []
#     for i in range(int(360/delta_rotation)):
#         degrees = int(i * delta_rotation)
#         read_file_name = "draw_test_"+str(degrees)+".jpg"
#         input_image = cv2.imread(read_file_name, cv2.IMREAD_GRAYSCALE)
#         ret, thresh_list = cv2.threshold(input_image,127,255,0)
#         n_white_pix2 = np.sum(thresh_list == 255)
#         #n_white_pix2.append(n_white_pix2)
#         #print(n_white_pix2)
#         intensity_ratio_list.append(n_white_pix2)
#     print(intensity_ratio_list)
            
#     #pixel_port.show()

    
#     # Calculate the intensity ratio
#     int_ratio = intensity_ratio_list * (1/n_white_pix)
#     #intensity_ratio = float((n_white_pix2/n_white_pix) * 100)
#     #print('Intensity Ratio per frame:{}%'.format(intensity_ratio))
#     #print(n_white_pix2)



def get_ceiling_plot(base_image, frames, frames_rand):
    total_pixel_list = get_list_of_total_of_shown_pixels(base_image)
    plt.style.use('dark_background')
    fig_size_h = 7
    fig_size_w = 7
    rain_drops = []
    n_drops = 0
    delay_in_ms = 10
    intensity = 15
    pixel_distance = 0.02
    

    # print(random_map_np)
    # Create new Figure and an Axes which fills it.
    fig = plt.figure(figsize=(fig_size_h, fig_size_w))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('equal')
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])

    # Fiber data
    n_drops = len(total_pixel_list)
    n_drops = int(math.sqrt(n_drops) * int(math.sqrt(n_drops)))
    # will lose some pixels due to uneven sqrt

    rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
                                        ('size',     float, 1),
                                        ('growth',   float, 1),
                                        ('color',    float, 4),])

    # Initialize the raindrops in random positions and with
    # random growth rates.
    # rain_drops['position'] = [0.5,0.5]
     # rain_drops['position'] = [5,5]
    # np.random.uniform(0, 1, (n_drops, 2))
    rain_drops['growth'] = 1/10
    # np.random.uniform(50, 200, n_drops)

    rain_drops['color'] = (1, 1, 1, 1)


    # Construct the scatter which we will update during animation
    # as the raindrops develop.
    scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
                    s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
                    facecolors='white')

    # scat = ax.scatter(random_map_np[:, 0], random_map_np[:, 1],
    #                 s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
    #                 facecolors='white')
    coord = []
    for x in range(53):
        for y in range(53):
            coord.append([pixel_distance * (x + 1), pixel_distance * (y + 1)])
            # coord=random_map_np

    # def update(frame_number):
    def update(frame_number):
        frame_index = frame_number % len(frames_rand)
        current_frame = frames_rand[frame_index]
        for n_drop in range(n_drops):
            coded_index = total_pixels_list[n_drop]
            frame_w = len(current_frame[0])
            frame_h = len(current_frame)
            rand_x_point , rand_y_point = get_decoded_coords(coded_index,frame_w_frame_h)
            dot_size = rain_drops['size'][n_drop] - int(intensity/2)

            if current_frame[rand_x_point, rand_y_point] != 0:
                dot_size = 1 * intensity
            
            rain_drops['size'][n_drop] = dot_size
            rain_drops['position'][n_drop] = coord[n_drop]
            rain_drops['color'][n_drop] = (1, 1, 1, 1)
            rain_drops['growth'][n_drop] = intensity/2

        # # print(frame_number)
        # # Get an index which we can use to re-spawn the oldest raindrop.
        # # current_index = timeframe % n_drops
        # current_index = timeframe % len(frames_rand)

        # # Make all colors more transparent as time progresses.
        # #rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
        # #rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)

        # # Make all circles bigger.
        # rain_drops['size'] -= rain_drops['growth']

        # # Pick a new position for oldest rain drop, resetting its size,
        # # color and growth factor.
        # # rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
        # # print(np.random.uniform(0, 1, 2))

        # # Evenly distribute fibers across x and y 
        # rain_drops['position'][current_index] = coord[current_index]
        # # rain_drops['position'][current_index] = random_map_np
        # rain_drops['size'][current_index] = 5
        # # rain_drops['color'][current_index] = (0, 0, 0, 1)
        # rain_drops['color'][current_index] = (1, 1, 1, 1)
        # # rain_drops['growth'][current_index] = 150.0
        # #print(np.random.uniform(50, 200))

        # Update the scatter collection, with the new colors, sizes and positions.
        scat.set_edgecolors(rain_drops['color'])
        scat.set_sizes(rain_drops['size'])
        scat.set_offsets(rain_drops['position'])
        

    # Construct the animation, using the update function as the animation
    # director.
    animation = FuncAnimation(fig, update, interval=delay_in_ms)
    plt.show()

#get coordinates for rand_x and rand_y points
def get_decoded_coords(coded_index,frame_w_frame_h):
    y_point = int(coded_index % image_h)
    x_point = int((coded_index - y_point) % image_w)
    return x_point, y_point

def get_encoded_coord(x_point, y_point, image_viewport_w, image_viewport_h):
    encoded_count = (x_point * image_viewport_h) + y_point
    return encoded_count


def get_intensity_ratio_plot():
    frames, frames_rand, intensity_ratio = main()
    x = []
    for i in range(int(360/delta_rotation)):
        degrees = int(i * delta_rotation)
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





main()
# get_intensity_ratio_plot()
get_ceiling_plot()

plt.show()

