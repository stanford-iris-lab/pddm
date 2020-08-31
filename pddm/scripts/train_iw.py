import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import numpy.random as npr
import tensorflow as tf
import pickle
import sys
import argparse
import traceback
import wandb


# my imports
from pddm.policies.policy_random import Policy_Random
from pddm.utils.helper_funcs import *
from pddm.regressors.dynamics_model import Dyn_Model
from pddm.regressors.discriminator_module import Discriminator
from pddm.policies.mpc_rollout import MPCRollout
from pddm.utils.loader import Loader
from pddm.utils.saver import Saver
from pddm.utils.data_processor import DataProcessor
from pddm.utils.data_structures import *
from pddm.utils.convert_to_parser_args import convert_to_parser_args
from pddm.utils import config_reader

SCRIPT_DIR = os.path.dirname(__file__)


def run_job(args, save_dir=None):

    if args.wandb:
        wandb.init(project="pddm-rollouts")


    # Continue training from an existing iteration
    if args.continue_run > -1:
        save_dir = os.path.join(SCRIPT_DIR, args.continue_run_filepath)

    tf.reset_default_graph()
    with tf.Session(config=get_gpu_config(args.use_gpu, args.gpu_frac)) as sess:

        ###########################################################################################################################################
        ############################################################## Initialization #############################################################
        ###########################################################################################################################################


        ############################################################
        ########## initialize some parameters (from args) ##########
        ############################################################

        env_name = args.env_name
        continue_run = args.continue_run
        K = args.K
        num_iters = args.num_iters
        num_trajectories_per_iter = args.num_trajectories_per_iter
        horizon = args.horizon

        # Set seeds
        npr.seed(args.seed)
        tf.set_random_seed(args.seed)

        ############################################################
        ###################### hardcoded args ######################
        ############################################################

        # Data types
        args.tf_datatype = tf.float32
        args.np_datatype = np.float32

        # Supervised learning noise, added to the training dataset
        args.noiseToSignal = 0.01

        # These are for *during* MPC rollouts,
        # they allow you to run the H-step candidate actions on the real dynamics
        # and compare the model's predicted outcomes vs. the true outcomes
        execute_sideRollouts = False
        plot_sideRollouts = True

        ############################################################
        ############ create loader, env and rand policy ############
        ############################################################

        loader = Loader(save_dir)
        env, dt_from_xml = create_env(env_name)
        args.dt_from_xml = dt_from_xml
        random_policy = Policy_Random(env.env)

        # doing a render here somehow allows it to not produce 
        # a seg fault error later when visualizing.
        if args.visualize_MPC_rollout:
            render_env(env)
            render_stop(env)

        ############################################################
        ################## Initialize or Load Run ##################
        ############################################################

        # Check whether each datapoint should be duplicated. 
        # e.g., for baoding, since ballA/B are interchangeable, we store as 2 different points.
        if "duplicateData_switchObjs" in dir(env.unwrapped_env):
            duplicateData_switchObjs = True
            indices_for_switching = [
                env.unwrapped_env.objInfo_start1,
                env.unwrapped_env.objInfo_start2,
                env.unwrapped_env.targetInfo_start1,
                env.unwrapped_env.targetInfo_start2,
            ]
        else:
            duplicateData_switchObjs = False
            indices_for_switching = []

        # initialize data processor
        data_processor = DataProcessor(
            args, duplicateData_switchObjs, indices_for_switching
        )

        disc_data_processor = DataProcessor(
            args, duplicateData_switchObjs, indices_for_switching
        )

        # start a fresh run
        if continue_run == -1:

            # random training/validation data
            if args.load_existing_random_data:
                raise NotImplementedError
                rollouts_trainRand, rollouts_valRand = loader.load_initialData()
            else:
                # training
                rollouts_trainRand = collect_random_rollouts(
                    env,
                    random_policy,
                    args.num_rand_rollouts_train,
                    args.rand_rollout_length,
                    dt_from_xml,
                    args,
                )
                # validation
                rollouts_valRand = collect_random_rollouts(
                    env,
                    random_policy,
                    args.num_rand_rollouts_val,
                    args.rand_rollout_length,
                    dt_from_xml,
                    args,
                )

            # convert (rollouts --> dataset)
            dataset_trainRand = data_processor.convertRolloutsToDatasets(
                rollouts_trainRand
            )
            dataset_valRand = data_processor.convertRolloutsToDatasets(rollouts_valRand)

            # onPol train/val data
            dataset_trainOnPol = Dataset()
            rollouts_trainOnPol = []
            rollouts_valOnPol = []

            # onPol disc train/val data
            disc_trainOnPol_real = []
            disc_trainOnPol_pred = []
            disc_valOnPol_real = []
            disc_valOnPol_pred = []


            # lists for saving
            trainingLoss_perIter = []
            rew_perIter = []
            scores_perIter = []
            trainingData_perIter = []

            # initialize counter
            counter = 0

        # continue from an existing run
        else:
            raise NotImplementedError
            # load data
            iter_data = loader.load_iter(continue_run - 1)

            # random data
            rollouts_trainRand, rollouts_valRand = loader.load_initialData()

            # onPol data
            rollouts_trainOnPol = iter_data.train_rollouts_onPol
            rollouts_valOnPol = iter_data.val_rollouts_onPol

            # convert (rollouts --> dataset)
            dataset_trainRand = data_processor.convertRolloutsToDatasets(
                rollouts_trainRand
            )
            dataset_valRand = data_processor.convertRolloutsToDatasets(rollouts_valRand)

            # lists for saving
            trainingLoss_perIter = iter_data.training_losses
            rew_perIter = iter_data.rollouts_rewardsPerIter
            scores_perIter = iter_data.rollouts_scoresPerIter
            trainingData_perIter = iter_data.training_numData

            # initialize counter
            counter = continue_run
            # how many iters to train for
            num_iters += continue_run

        ### check data dims
        inputSize, outputSize, acSize = check_dims(dataset_trainRand, env)

        ### amount of data
        numData_train_rand = get_num_data(rollouts_trainRand)

        #############################################################
        ############### Dynamics Model and Controller ###############
        #############################################################

        dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=args)

        discriminator = Discriminator(inputSize*2 - acSize, acSize, sess, params=args)
        
        
        mpc_rollout = MPCRollout(
            env,
            dyn_models,
            random_policy,
            execute_sideRollouts,
            plot_sideRollouts,
            args,
            discriminator = discriminator
        )

        # Initialize TF variables
        sess.run(tf.global_variables_initializer())

        # Initialiaze saver
        saver = Saver(save_dir, sess)
        saver.save_initialData(args, rollouts_trainRand, rollouts_valRand)


        firstTime = True
        rollouts_info_prevIter, list_mpes, list_scores, list_rewards = (
            None,
            None,
            None,
            None,
        )

        ###########################################################################################################################################
        ############################################################## THE MAIN LOOP ##############################################################
        ###########################################################################################################################################
        while counter < num_iters:

            ##############################################################
            ########## Initialize iteration and preprocess data ##########
            ##############################################################

            # init vars for this iteration
            saver_data = DataPerIter()
            saver.iter_num = counter

            # onPolicy validation doesn't exist yet, so just make it same as rand validation
            if counter == 0:
                rollouts_valOnPol = rollouts_valRand

            # convert (rollouts --> dataset)
            dataset_trainOnPol = data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)
            dataset_valOnPol = data_processor.convertRolloutsToDatasets(rollouts_valOnPol)

            # amount of data
            numData_train_onPol = get_num_data(rollouts_trainOnPol)

            # calculate mean/std of all data
            data_processor.update_stats(
                dyn_models, dataset_trainRand, dataset_trainOnPol
            )
         
            # preprocess datasets to mean0/std1 + clip actions
            preprocessed_data_trainRand = data_processor.preprocess_data(dataset_trainRand)
            preprocessed_data_valRand = data_processor.preprocess_data(dataset_valRand)
            preprocessed_data_trainOnPol = data_processor.preprocess_data(dataset_trainOnPol)
            preprocessed_data_valOnPol = data_processor.preprocess_data(dataset_valOnPol)

            # convert datasets (x,y,z) --> training sets ((state,action), (next_state))
            inputs, outputs = data_processor.xyz_to_inpOutp(preprocessed_data_trainRand)
            inputs_val, outputs_val = data_processor.xyz_to_inpOutp(preprocessed_data_valRand)
            inputs_onPol, outputs_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_trainOnPol)
            inputs_val_onPol, outputs_val_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_valOnPol)

            #############################################################
            ###################### Train the model ######################
            #############################################################

            if not (args.print_minimal):
                print("\n#####################################")
                print("Training the dynamics model..... iteration ", counter)
                print("#####################################\n")
                print("    amount of random data: ", numData_train_rand)
                print("    amount of onPol data: ", numData_train_onPol)

            """Why?!"""
            ### copy train_onPol until it's big enough
            if len(inputs_onPol) > 0:
                while inputs_onPol.shape[0] < inputs.shape[0]:
                    inputs_onPol = np.concatenate([inputs_onPol, inputs_onPol])
                    outputs_onPol = np.concatenate([outputs_onPol, outputs_onPol])

            """Why?!"""
            ### copy val_onPol until it's big enough
            while inputs_val_onPol.shape[0] < args.batchsize:
                inputs_val_onPol = np.concatenate(
                    [inputs_val_onPol, inputs_val_onPol], 0
                )
                outputs_val_onPol = np.concatenate(
                    [outputs_val_onPol, outputs_val_onPol], 0
                )

            # restore model if doing continue_run otherwise
            # re-initialize all vars (randomly) if training from scratch
            if args.warmstart_training:
                if firstTime:
                    if continue_run > 0:
                        restore_path = (
                            save_dir
                            + "/models/model_aggIter"
                            + str(continue_run - 1)
                            + ".ckpt"
                        )
                        saver.tf_saver.restore(sess, restore_path)
                        print("\n\nModel restored from ", restore_path, "\n\n")
            else:
                sess.run(tf.global_variables_initializer())

            # number of training epochs
            if counter == 0:
                nEpoch_use = args.nEpoch_init
            else:
                nEpoch_use = args.nEpoch

            # restore or train model.
            # TODO: add disc to loading mode.
            if args.always_use_savedModel:
                raise NotImplementedError
            else:
                # Train model 
                ## Training on random and on policy data. 
                training_loss, training_lists_to_save = dyn_models.train(
                    inputs,
                    outputs,
                    inputs_onPol,
                    outputs_onPol,
                    nEpoch_use,
                    inputs_val=inputs_val,
                    outputs_val=outputs_val,
                    inputs_val_onPol=inputs_val_onPol,
                    outputs_val_onPol=outputs_val_onPol,
                    wandb=wandb
                )

            # saving rollout info
            rollouts_info = []
            list_rewards = []
            list_scores = []
            list_mpes = []

            if not args.print_minimal:
                print("\n#####################################")
                print("performing on-policy MPC rollouts... iter ", counter)
                print("#####################################\n")

            ###############################################################
            ################# Perform Rollouts with Model #################
            ###############################################################

            for rollout_num in range(num_trajectories_per_iter):
                # perform 1 MPC rollout 
                if not args.print_minimal:
                    print(
                        "\n####################### Performing MPC rollout [Model Only] #",
                        rollout_num,
                    )

                # reset env
                starting_observation, starting_state = env.reset(
                    return_start_state=True
                )

                rollout_info = mpc_rollout.perform_rollout(
                    starting_state,
                    starting_observation,
                    controller_type=args.controller_type,
                    take_exploratory_actions=False,
                )

                if args.wandb:
                    wandb.log({"model_only/rollout_reward": rollout_info['rollout_rewardTotal']})
                

                """
                NOTE: can sometimes set take_exploratory_actions=True
                in order to use ensemble disagreement for exploration
                """
                # save rollout info (if long enough) 
                
                if len(rollout_info["observations"]) > K:
                    list_rewards.append(rollout_info["rollout_rewardTotal"])
                    list_scores.append(rollout_info["rollout_meanFinalScore"])
                    list_mpes.append(np.mean(rollout_info["mpe_1step"]))
                    rollouts_info.append(rollout_info)

            rollouts_info_prevIter = rollouts_info.copy()

            # visualize rollout
            if args.visualize_MPC_rollout:
                print("\n\nPAUSED FOR VISUALIZATION. Continue when ready to visualize.")
                import IPython

                IPython.embed()
                for vis_index in range(len(rollouts_info)):
                    visualize_rendering(rollouts_info[vis_index], env, args)

            #############################################################
            ##### aggregate some random rollouts into training data #####
            #############################################################

            num_rand_rollouts = 5
            rollouts_rand = []

            # random sampling params
            random_sampling_params = dict(
                sample_velocities=args.rand_policy_sample_velocities,
                vel_min=args.rand_policy_vel_min,
                vel_max=args.rand_policy_vel_max,
                hold_action=args.rand_policy_hold_action,
            )


            # Do rollouts
            for rand_rollout_counter in range(num_rand_rollouts):
                # reset env
                starting_observation, starting_state = env.reset(
                    return_start_state=True
                )

                rollout_info = mpc_rollout.perform_rollout(
                    starting_state,
                    starting_observation,
                    controller_type=args.controller_type,
                    take_exploratory_actions=False,
                    use_disc=False,
                    isRandom=True,
                    random_sampling_params=random_sampling_params
                )

                rollout = Rollout(
                    rollout_info["observations"],
                    rollout_info["actions"],
                    rollout_info["rollout_rewardTotal"],
                    rollout_info["starting_state"]
                )

                rollouts_rand.append(rollout)

            # convert (rollouts --> dataset)
            dataset_rand_new = data_processor.convertRolloutsToDatasets(rollouts_rand)

            # concat this dataset with the existing dataset_trainRand
            dataset_trainRand = concat_datasets(dataset_trainRand, dataset_rand_new)

            #############################################################
            ########### aggregate MPC rollouts into train/val ###########
            #############################################################

            num_mpc_rollouts = len(rollouts_info)
            rollouts_train = []
            rollouts_val = []

            for i in range(num_mpc_rollouts):
                rollout = Rollout(
                    rollouts_info[i]["observations"],
                    rollouts_info[i]["actions"],
                    rollouts_info[i]["rollout_rewardTotal"],
                    rollouts_info[i]["starting_state"],
                )

                if i < int(num_mpc_rollouts * 0.9):
                    rollouts_train.append(rollout)
                else:
                    rollouts_val.append(rollout)

            """Create disc data"""
            disc_real_transitions_train = []
            disc_real_transitions_val = []
            disc_pred_transitions_train = []
            disc_pred_transitions_val = []

            for i in range(num_mpc_rollouts):
                for t in range(rollouts_info[i]['actions'].shape[0]):
                    # model predicts diffs so need to calcuate it
                    real_transition = Transition(
                        state = rollouts_info[i]['observations'][t,:], 
                        action = rollouts_info[i]['actions'][t, :],
                        difference = rollouts_info[i]['observations'][t + 1,:] - rollouts_info[i]['observations'][t,:]
                    )

                    pred_transition = Transition(
                        state = rollouts_info[i]['observations'][t,:], 
                        action = rollouts_info[i]['actions'][t, :], 
                        difference = rollouts_info[i]['predicted_next_state'][t,:]
                    )
                    
                    if i < int(num_mpc_rollouts * 0.9):
                        disc_real_transitions_train.append(real_transition)
                        disc_pred_transitions_train.append(pred_transition)
                    else:
                        disc_real_transitions_val.append(real_transition)
                        disc_pred_transitions_val.append(pred_transition)

            # aggregate into training data
            if counter == 0:
                rollouts_valOnPol = []
            rollouts_trainOnPol = rollouts_trainOnPol + rollouts_train
            rollouts_valOnPol = rollouts_valOnPol + rollouts_val

            # aggregate into disc training data
            if counter == 0:
                disc_valOnPol_real = []
                disc_valOnPol_pred = []
            
            disc_trainOnPol_real = disc_trainOnPol_real + disc_real_transitions_train
            disc_trainOnPol_pred = disc_trainOnPol_pred + disc_pred_transitions_train
            disc_valOnPol_real = disc_valOnPol_real + disc_real_transitions_val
            disc_valOnPol_pred = disc_valOnPol_pred + disc_pred_transitions_val

            #########################################################
            ### save everything about this iter of model training ###
            """do we need to save disc stuff?"""
            trainingData_perIter.append(numData_train_rand + numData_train_onPol)
            trainingLoss_perIter.append(training_loss)

            ### stage relevant info for saving
            saver_data.training_numData = trainingData_perIter
            saver_data.training_losses = trainingLoss_perIter
            saver_data.training_lists_to_save = training_lists_to_save
            # Note: the on-policy rollouts include curr iter's rollouts
            # (so next iter can be directly trained on these)
            saver_data.train_rollouts_onPol = rollouts_trainOnPol
            saver_data.val_rollouts_onPol = rollouts_valOnPol
            saver_data.normalization_data = data_processor.get_normalization_data()
            saver_data.counter = counter

            ### save all info from this training iteration
            saver.save_model()
            saver.save_training_info(saver_data)
            ###                                                 ###
            #######################################################

            #######################################################
            ### save everything about this iter of MPC rollouts ###

            # append onto rewards/scores
            rew_perIter.append([np.mean(list_rewards), np.std(list_rewards)])
            scores_perIter.append([np.mean(list_scores), np.std(list_scores)])
            # save
            saver_data.rollouts_rewardsPerIter = rew_perIter
            saver_data.rollouts_scoresPerIter = scores_perIter
            saver_data.rollouts_info = rollouts_info
            saver.save_rollout_info(saver_data)
            counter = counter + 1

            ###                                                 ###
            #######################################################

            firstTime = False

            """ ################################################################################################################################## """
            """ ######################################################## Second Main Loop ######################################################## """
            """ ################################################################################################################################## """


            #############################################################
            ########## Initialize iteration && preprocess data ##########
            #############################################################

            # init vars for this iteration
            saver_data = DataPerIter()
            saver.iter_num = counter

            # onPolicy validation doesn't exist yet, so just make it same as rand validation
            if counter == 0:
                rollouts_valOnPol = rollouts_valRand

            # convert (rollouts --> dataset)
            # dataset_trainOnPol = disc_data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)
            # dataset_valOnPol = disc_data_processor.convertRolloutsToDatasets(rollouts_valOnPol)

            disc_dataset_trainOnPol_real = disc_data_processor.convertTransitionsToDatasets(disc_trainOnPol_real)
            disc_dataset_trainOnPol_pred = disc_data_processor.convertTransitionsToDatasets(disc_trainOnPol_pred)
            disc_dataset_valOnPol_real = disc_data_processor.convertTransitionsToDatasets(disc_valOnPol_real)
            disc_dataset_valOnPol_pred = disc_data_processor.convertTransitionsToDatasets(disc_valOnPol_pred)
            

            # amount of data
            numData_train_onPol = get_num_data(rollouts_trainOnPol)

            # calculate mean/std of all data
            # TODO: fix rand
            disc_data_processor.update_stats(
                discriminator, disc_dataset_trainOnPol_real, disc_dataset_trainOnPol_pred
            )

            # preprocess datasets to mean0/std1 + clip actions
            # preprocessed_data_trainRand = disc_data_processor.preprocess_data(dataset_trainRand)
            # preprocessed_data_valRand = disc_data_processor.preprocess_data(dataset_valRand)
            preprocessed_disc_data_trainOnPol_real = disc_data_processor.preprocess_data(disc_dataset_trainOnPol_real)
            preprocessed_disc_data_trainOnPol_pred = disc_data_processor.preprocess_data(disc_dataset_trainOnPol_pred)
            
            preprocessed_disc_data_valOnPol_real = disc_data_processor.preprocess_data(disc_dataset_valOnPol_real)
            preprocessed_disc_data_valOnPol_pred = disc_data_processor.preprocess_data(disc_dataset_valOnPol_pred)
            
            # preprocessed_data_valOnPol = disc_data_processor.preprocess_data(dataset_valOnPol)

            # convert datasets (x,y,z) --> training sets ((state,action), (next_state))
            # inputs, outputs = disc_data_processor.xyz_to_inpOutp(preprocessed_data_trainRand)
            # inputs_val, outputs_val = disc_data_processor.xyz_to_inpOutp(preprocessed_data_valRand)
            
            disc_inputs_onPol, disc_outputs_onPol = disc_data_processor.xyz_to_disc_io(preprocessed_disc_data_trainOnPol_real, preprocessed_disc_data_trainOnPol_pred)
            disc_inputs_val_onPol, disc_outputs_val_onPol = disc_data_processor.xyz_to_disc_io(preprocessed_disc_data_valOnPol_real, preprocessed_disc_data_valOnPol_pred)

            #############################################################
            ###################### Train the model ######################
            #############################################################

            if not (args.print_minimal):
                print("\n#####################################")
                print("Training the discriminator ..... iteration ", counter)
                print("#####################################\n")
                print("    amount of random data: ", numData_train_rand)
                print("    amount of onPol data: ", numData_train_onPol)

            print(f"before: inputs_onPol.shape: {disc_inputs_onPol.shape}")
            """Why?!"""
            ### copy train_onPol until it's big enough
            if len(disc_inputs_onPol) > 0:
                while disc_inputs_onPol.shape[0] < inputs.shape[0]:
                    disc_inputs_onPol = np.concatenate([disc_inputs_onPol, disc_inputs_onPol])
                    disc_outputs_onPol = np.concatenate([disc_outputs_onPol, disc_outputs_onPol])

            """Why?!"""
            ### copy val_onPol until it's big enough
            while disc_inputs_val_onPol.shape[0] < args.batchsize:
                disc_inputs_val_onPol = np.concatenate(
                    [disc_inputs_val_onPol, disc_inputs_val_onPol], 0
                )
                disc_outputs_val_onPol = np.concatenate(
                    [disc_outputs_val_onPol, disc_outputs_val_onPol], 0
                )

            print(f"after: inputs_onPol.shape: {disc_inputs_onPol.shape}")
            
            # restore model if doing continue_run otherwise re-initialize all vars (randomly) 
            if args.warmstart_training:
                if firstTime:
                    if continue_run > 0:
                        restore_path = (
                            save_dir
                            + "/models/model_aggIter"
                            + str(continue_run - 1)
                            + ".ckpt"
                        )
                        saver.tf_saver.restore(sess, restore_path)
                        print("\n\nModel restored from ", restore_path, "\n\n")
            else:
                sess.run(tf.global_variables_initializer())

            # number of training epochs
            if counter == 0:
                nEpoch_use = args.nEpoch_init
            else:
                nEpoch_use = args.nEpoch

            # restore or train model.
            if args.always_use_savedModel:
                if continue_run > 0:
                    restore_path = (
                        save_dir
                        + "/models/model_aggIter"
                        + str(continue_run - 1)
                        + ".ckpt"
                    )
                else:
                    restore_path = save_dir + "/models/finalModel.ckpt"

                saver.tf_saver.restore(sess, restore_path)
                print("\n\nModel restored from ", restore_path, "\n\n")

                # empty vars, for saving
                training_loss = 0
                training_lists_to_save = dict(
                    training_loss_list=0,
                    val_loss_list_rand=0,
                    val_loss_list_onPol=0,
                    val_loss_list_xaxis=0,
                    rand_loss_list=0,
                    onPol_loss_list=0,
                )
            else:
                # Train model on random and on policy data. 
                """What is this expansion for?"""
                disc_inputs_onPol = np.expand_dims(disc_inputs_onPol, axis = 1)
                disc_outputs_onPol = np.expand_dims(disc_outputs_onPol, axis = 1)

                disc_inputs_val_onPol = np.expand_dims(disc_inputs_val_onPol, axis = 1)
                disc_outputs_val_onPol = np.expand_dims(disc_outputs_val_onPol, axis = 1)

                # TODO: handle random data inputs
                training_loss, training_lists_to_save = discriminator.train(
                    inputs,
                    outputs,
                    disc_inputs_onPol,
                    disc_outputs_onPol,
                    nEpoch_use,
                    inputs_val=inputs_val,
                    outputs_val=outputs_val,
                    inputs_val_onPol=disc_inputs_val_onPol,
                    outputs_val_onPol=disc_outputs_val_onPol,
                    wandb=wandb
                )
            
            # saving rollout info
            rollouts_info = []
            list_rewards = []
            list_scores = []
            list_mpes = []

            if not args.print_minimal:
                print("\n#####################################")
                print("performing on-policy MPC rollouts... iter ", counter)
                print("#####################################\n")

            ##############################################################
            ###################### Perform Rollouts ######################
            ##############################################################
            for rollout_num in range(num_trajectories_per_iter):

                # perform 1 MPC rollout 
                if not args.print_minimal:
                    print(
                        "\n####################### Performing MPC rollout [Model + Disc] #",
                        rollout_num,
                    )

                # reset env
                starting_observation, starting_state = env.reset(
                    return_start_state=True
                )

                rollout_info = mpc_rollout.perform_rollout(
                    starting_state,
                    starting_observation,
                    controller_type=args.controller_type,
                    take_exploratory_actions=False,
                    use_disc=True
                )

                if args.wandb:
                    wandb.log({"model_disc/rollout_reward": rollout_info['rollout_rewardTotal']})

                """
                NOTE: can sometimes set take_exploratory_actions=True
                in order to use ensemble disagreement for exploration
                """
                # save rollout info (if long enough) 
                
                if len(rollout_info["observations"]) > K:
                    list_rewards.append(rollout_info["rollout_rewardTotal"])
                    list_scores.append(rollout_info["rollout_meanFinalScore"])
                    list_mpes.append(np.mean(rollout_info["mpe_1step"]))
                    rollouts_info.append(rollout_info)

            rollouts_info_prevIter = rollouts_info.copy()

            # visualize rollout
            if args.visualize_MPC_rollout:
                print("\n\nPAUSED FOR VISUALIZATION. Continue when ready to visualize.")
                import IPython

                IPython.embed()
                for vis_index in range(len(rollouts_info)):
                    visualize_rendering(rollouts_info[vis_index], env, args)

            #############################################################
            ##### aggregate some random rollouts into training data #####
            #############################################################

            num_rand_rollouts = 5


            rollouts_rand = []
            # random sampling params
            random_sampling_params = dict(
                sample_velocities=args.rand_policy_sample_velocities,
                vel_min=args.rand_policy_vel_min,
                vel_max=args.rand_policy_vel_max,
                hold_action=args.rand_policy_hold_action,
            )

            for rand_rollout_counter in range(num_rand_rollouts):
                # reset env
                starting_observation, starting_state = env.reset(
                    return_start_state=True
                )

                rollout_info = mpc_rollout.perform_rollout(
                    starting_state,
                    starting_observation,
                    controller_type=args.controller_type,
                    take_exploratory_actions=False,
                    use_disc=False,
                    isRandom=True,
                    random_sampling_params=random_sampling_params
                )

                rollout = Rollout(
                    rollout_info["observations"],
                    rollout_info["actions"],
                    rollout_info["rollout_rewardTotal"],
                    rollout_info["starting_state"]
                )

                rollouts_rand.append(rollout)



            # convert (rollouts --> dataset)
            dataset_rand_new = disc_data_processor.convertRolloutsToDatasets(rollouts_rand)

            # concat this dataset with the existing dataset_trainRand
            dataset_trainRand = concat_datasets(dataset_trainRand, dataset_rand_new)

            #############################################################
            ########### aggregate MPC rollouts into train/val ###########
            #############################################################

            num_mpc_rollouts = len(rollouts_info)
            rollouts_train = []
            rollouts_val = []

            for i in range(num_mpc_rollouts):
                rollout = Rollout(
                    rollouts_info[i]["observations"],
                    rollouts_info[i]["actions"],
                    rollouts_info[i]["rollout_rewardTotal"],
                    rollouts_info[i]["starting_state"],
                )

                if i < int(num_mpc_rollouts * 0.9):
                    rollouts_train.append(rollout)
                else:
                    rollouts_val.append(rollout)

            # aggregate into training data
            if counter == 0:
                rollouts_valOnPol = []
            rollouts_trainOnPol = rollouts_trainOnPol + rollouts_train
            rollouts_valOnPol = rollouts_valOnPol + rollouts_val

            #############################################################
            ######## save everything about this iter of training ########
            #############################################################

            trainingData_perIter.append(numData_train_rand + numData_train_onPol)
            trainingLoss_perIter.append(training_loss)

            ### stage relevant info for saving
            saver_data.training_numData = trainingData_perIter
            saver_data.training_losses = trainingLoss_perIter
            saver_data.training_lists_to_save = training_lists_to_save
            # Note: the on-policy rollouts include curr iter's rollouts
            # (so next iter can be directly trained on these)
            saver_data.train_rollouts_onPol = rollouts_trainOnPol
            saver_data.val_rollouts_onPol = rollouts_valOnPol
            saver_data.normalization_data = disc_data_processor.get_normalization_data()
            saver_data.counter = counter

            ### save all info from this training iteration
            saver.save_model()
            saver.save_training_info(saver_data)
            

            #############################################################
            ###### save everything about this iter of MPC rollouts ######
            #############################################################

            # append onto rewards/scores
            rew_perIter.append([np.mean(list_rewards), np.std(list_rewards)])
            scores_perIter.append([np.mean(list_scores), np.std(list_scores)])
            # save
            saver_data.rollouts_rewardsPerIter = rew_perIter
            saver_data.rollouts_scoresPerIter = scores_perIter
            saver_data.rollouts_info = rollouts_info
            saver.save_rollout_info(saver_data)
            counter = counter + 1

            firstTime = False
        return

        ###########################################################################################################################################
        ############################################################ End of MAIN LOOP #############################################################
        ###########################################################################################################################################


def main():

    #####################
    # training args
    #####################

    parser = argparse.ArgumentParser(
        # Show default value in the help doc.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        nargs="*",
        help=(
            "Path to the job data config file. This is specified relative "
            "to working directory"
        ),
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="output",
        help=(
            "Directory to output trained policies, logs, and plots. A subdirectory "
            "is created for each job. This is speficified relative to  "
            "working directory"
        ),
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("-frac", "--gpu_frac", type=float, default=0.9)
    general_args = parser.parse_args()

    

    #####################
    # job configs
    #####################

    # Get the job config files
    jobs = config_reader.process_config_files(general_args.config)
    assert jobs, "No jobs found from config."

    # Create the output directory if not present.
    output_dir = general_args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    # Run separate experiment for each variant in the config
    for index, job in enumerate(jobs):

        # add an index to jobname, if there is more than 1 job
        if len(jobs) > 1:
            job["job_name"] = "{}_{}".format(job["job_name"], index)

        # convert job dictionary to different format
        args_list = config_dict_to_flags(job)
        args = convert_to_parser_args(args_list)

        # copy some general_args into args
        args.use_gpu = general_args.use_gpu
        args.gpu_frac = general_args.gpu_frac
        args.wandb = general_args.wandb

        # directory name for this experiment
        job["output_dir"] = os.path.join(output_dir, job["job_name"])

        ################
        ### run job
        ################

        try:
            run_job(args, job["output_dir"])
        except (KeyboardInterrupt, SystemExit):
            print("Terminating...")
            sys.exit(0)
        except Exception as e:
            print("ERROR: Exception occured while running a job....")
            traceback.print_exc()


if __name__ == "__main__":
    main()
